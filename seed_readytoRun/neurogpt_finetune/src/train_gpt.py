#!/usr/bin/env python3

"""
train.py

Training of models on given data. See get_args() for
details on command line arguments.

To train a model, multiple core components from ..src/
are invoked:

src/batcher: Building PyTorch dataloaders for given data.
src/embedder: Embedding of inputs into embedding space,
    training-style specific addition of training tokens
    and masking, and computation of training-style specific
    losses.
    Valid training styles:
        - CSM (Causal Sequence Modeling)
        - decoding
src/decoder: Model architecture used for decoding / sequence modeling.
    One of the following:
        - GPT
        - PretrainedBERT (as provided by HuggingFace)
src/unembedder: Projecting sequence output of decoder back
    to input space.
src/trainer: Trainer for model; invokes instance of
    Hugging Face's Trainer object.
src/model: Build full model from components (ie., embedder,
    decoder, unembedder). See make_model() below for details.
"""
from batcher.downstream_dataset import MotorImageryDataset, KaggleERNDataset, Motor6ClassDataset, Emotion7ClassDataset
import torch
import sys
import os
import glob
import argparse
import pdb
from typing import Dict
import json
from datetime import datetime
from numpy import random
import pandas as pd
import numpy as np
from encoder.conformer_braindecode import EEGConformer
from torch import manual_seed

# Fix for torch._six compatibility issue with deepspeed
# This module was removed in newer PyTorch versions but deepspeed still tries to import it
# Must inject into sys.modules so that 'from torch._six import inf' works when deepspeed imports it
# This needs to be done as early as possible, before any imports that might trigger deepspeed
if not hasattr(torch, '_six'):
    class _Six:
        string_types = (str,)
        integer_types = (int,)
        text_type = str
        binary_type = bytes
        class_types = (type,)
        inf = float('inf')
    torch._six = _Six()
    # Inject into sys.modules so that 'from torch._six import inf' works
    # This is necessary because deepspeed does 'from torch._six import inf' at module level
    sys.modules['torch._six'] = torch._six

from utils import cv_split_bci, read_threshold_sub

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, "../"))
# from batcher.make import make_batcher
from batcher.base import EEGDataset
from decoder.make_decoder import make_decoder
from embedder.make import make_embedder
from trainer.make import make_trainer
from trainer.base import Trainer
from decoder.unembedder import make_unembedder

os.environ["WANDB_DISABLED"] = "true"


def train(config: Dict = None) -> Trainer:
    """Model training according to config.
    -> see get_args() below for all command
    line arguments.
    """

    if config is None:
        config = get_config()

    if config["do_train"]:
        os.makedirs(config["log_dir"], exist_ok=True)
        resume_path = (
            str(config["resume_from"]) if config["resume_from"] is not None else None
        )

        if resume_path is not None:
            config_filepath = os.path.join(config["resume_from"], "train_config.json")

            if os.path.isfile(config_filepath):
                print(f"Loading training config from {config_filepath}")

                with open(config_filepath, "r") as f:
                    config = json.load(f)

            else:

                with open(config_filepath, "w") as f:
                    json.dump(config, f, indent=2)

            checkpoints = [
                int(p.split("checkpoint-")[1])
                for p in os.listdir(resume_path)
                if "checkpoint-" in p and os.path.isdir(os.path.join(resume_path, p))
            ]
            last_checkpoint = max(checkpoints)
            print(
                f"Resuming training from checkpoint-{last_checkpoint} in {resume_path}"
            )
            config["resume_from"] = os.path.join(
                resume_path, f"checkpoint-{last_checkpoint}"
            )

        else:
            config_filepath = os.path.join(config["log_dir"], "train_config.json")

            with open(config_filepath, "w") as f:
                json.dump(config, f, indent=2)

            config["resume_from"] = None

    assert config["training_style"] in {
        "CSM",
        "CSM_causal",
        "decoding",
    }, f'{config["training_style"]} is not supported.'

    assert config["architecture"] in {
        "GPT",
        "PretrainedGPT2",
    }, f'{config["architecture"]} is not supported.'

    if config["set_seed"]:
        random.seed(config["seed"])
        manual_seed(config["seed"])

    # handles the input part, which are the output from encoder.
    if config["training_style"] == "decoding":
        dataset_name = config["dataset_name"]
        downstream_path = config["dst_data_path"]

        print(f"INFO: Using {dataset_name} for downstream task.")

        if dataset_name in ["KaggleERN", "stress"]:

            file_extension = "*.pickle"
            train_path = os.path.join(downstream_path, "train")
            val_path = os.path.join(downstream_path, "val")
            test_path = os.path.join(downstream_path, "test")
            train_files = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(train_path, file_extension))
            ]
            val_files = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(val_path, file_extension))
            ]
            test_files = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(test_path, file_extension))
            ]

            matrix_p_path = ""
            if dataset_name == "KaggleERN":
                matrix_p_path = "../inputs/tMatrix_22x56_KaggleERN.npy"
            elif dataset_name == "stress":
                matrix_p_path = "../inputs/tMatrix_22x30_stress.npy"
            else:
                raise ValueError(f"Undefined dataset: {dataset_name}")

            train_dataset = KaggleERNDataset(
                train_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=train_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

            validation_dataset = KaggleERNDataset(
                val_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=val_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

            test_dataset = KaggleERNDataset(
                test_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=test_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

        elif dataset_name == "bci2a":

            train_folds, test_folds = cv_split_bci(
                sorted(os.listdir(downstream_path))[:18]
            )
            train_files = train_folds[config["fold_i"]]
            test_files = test_folds[config["fold_i"]]

            train_dataset = MotorImageryDataset(
                train_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=downstream_path,
                gpt_only=not config["use_encoder"],
            )
            # pdb.set_trace()

            test_dataset = MotorImageryDataset(
                test_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=downstream_path,
                gpt_only=not config["use_encoder"],
            )

            validation_dataset = test_dataset
            test_dataset = train_dataset

        elif dataset_name == "motor6class":
            # 运动6分类数据集
            # 确保路径是绝对路径或相对于脚本目录的路径
            if not os.path.isabs(downstream_path):
                script_dir = os.path.dirname(os.path.realpath(__file__))
                script_dir = os.path.dirname(script_dir)  # 回到neurogpt_finetune目录
                # 处理相对路径：如果以 ../ 开头，从当前工作目录解析；否则从脚本目录解析
                if downstream_path.startswith('../'):
                    # 从当前工作目录解析相对路径
                    downstream_path = os.path.abspath(downstream_path)
                else:
                    # 从脚本目录解析相对路径
                    downstream_path = os.path.normpath(os.path.join(script_dir, downstream_path.lstrip('./')))
            
            file_extension = "*.pickle"
            train_path = os.path.join(downstream_path, "train")
            val_path = os.path.join(downstream_path, "val")
            test_path = os.path.join(downstream_path, "test")
            
            # 检查路径是否存在
            if not os.path.exists(train_path):
                raise ValueError(f"训练数据路径不存在: {train_path}")
            if not os.path.exists(val_path):
                raise ValueError(f"验证数据路径不存在: {val_path}")
            if not os.path.exists(test_path):
                raise ValueError(f"测试数据路径不存在: {test_path}")
            
            train_files = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(train_path, file_extension))
            ]
            val_files = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(val_path, file_extension))
            ]
            test_files = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(test_path, file_extension))
            ]
            
            print(f"INFO: 找到 {len(train_files)} 个训练文件, {len(val_files)} 个验证文件, {len(test_files)} 个测试文件")

            # 转换矩阵路径（相对于脚本运行目录）
            matrix_p_path = config.get("matrix_p_path", "tMatrix_22x20_motor.npy")
            if not os.path.isabs(matrix_p_path):
                # 如果是相对路径，先尝试当前路径是否存在
                if not os.path.exists(matrix_p_path):
                    # 如果以 ../ 开头，从当前工作目录解析
                    if matrix_p_path.startswith('../'):
                        matrix_p_path = os.path.abspath(matrix_p_path)
                    else:
                        # 否则从脚本目录解析
                        script_dir = os.path.dirname(os.path.realpath(__file__))
                        script_dir = os.path.dirname(script_dir)  # 回到neurogpt_finetune目录
                        matrix_p_path = os.path.join(script_dir, matrix_p_path.lstrip('./'))
            
            train_dataset = Motor6ClassDataset(
                train_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=train_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

            validation_dataset = Motor6ClassDataset(
                val_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=val_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

            test_dataset = Motor6ClassDataset(
                test_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=test_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

        elif dataset_name == "emotion7class":
            # Emotion 7分类数据集
            # 确保路径是绝对路径或相对于脚本目录的路径
            if not os.path.isabs(downstream_path):
                script_dir = os.path.dirname(os.path.realpath(__file__))
                script_dir = os.path.dirname(script_dir)  # 回到neurogpt_finetune目录
                # 处理相对路径：如果以 ../ 开头，从当前工作目录解析；否则从脚本目录解析
                if downstream_path.startswith('../'):
                    # 从当前工作目录解析相对路径
                    downstream_path = os.path.abspath(downstream_path)
                else:
                    # 从脚本目录解析相对路径
                    downstream_path = os.path.normpath(os.path.join(script_dir, downstream_path.lstrip('./')))
            
            # 数据在子文件夹内（subject_X），需要递归查找
            file_extension = "*.pickle"
            train_path = os.path.join(downstream_path, "train")
            val_path = os.path.join(downstream_path, "val")
            test_path = os.path.join(downstream_path, "test")
            
            # 检查路径是否存在
            if not os.path.exists(train_path):
                raise ValueError(f"训练数据路径不存在: {train_path}")
            if not os.path.exists(val_path):
                raise ValueError(f"验证数据路径不存在: {val_path}")
            if not os.path.exists(test_path):
                raise ValueError(f"测试数据路径不存在: {test_path}")
            
            # 递归查找所有pickle文件（包括子文件夹）
            train_files = [
                os.path.relpath(f, train_path)
                for f in glob.glob(os.path.join(train_path, "**", file_extension), recursive=True)
            ]
            val_files = [
                os.path.relpath(f, val_path)
                for f in glob.glob(os.path.join(val_path, "**", file_extension), recursive=True)
            ]
            test_files = [
                os.path.relpath(f, test_path)
                for f in glob.glob(os.path.join(test_path, "**", file_extension), recursive=True)
            ]
            
            print(f"INFO: 找到 {len(train_files)} 个训练文件, {len(val_files)} 个验证文件, {len(test_files)} 个测试文件")

            # 转换矩阵路径（相对于脚本运行目录）
            matrix_p_path = config.get("matrix_p_path", "tMatrix_22x62_seed.npy")
            if not os.path.isabs(matrix_p_path):
                # 如果是相对路径，先尝试当前路径是否存在
                if not os.path.exists(matrix_p_path):
                    # 如果以 ../ 开头，从当前工作目录解析
                    if matrix_p_path.startswith('../'):
                        matrix_p_path = os.path.abspath(matrix_p_path)
                    else:
                        # 否则从脚本目录解析
                        script_dir = os.path.dirname(os.path.realpath(__file__))
                        script_dir = os.path.dirname(script_dir)  # 回到neurogpt_finetune目录
                        matrix_p_path = os.path.join(script_dir, matrix_p_path.lstrip('./'))
            
            train_dataset = Emotion7ClassDataset(
                train_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=train_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

            validation_dataset = Emotion7ClassDataset(
                val_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=val_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

            test_dataset = Emotion7ClassDataset(
                test_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=test_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )
        else:
            # 如果 dataset_name 不匹配任何已知的情況
            raise ValueError(
                f"Unknown dataset_name: {dataset_name} for training_style 'decoding'. Please check your config."
            )

    else:
        root_path = config["train_data_path"]
        files = read_threshold_sub(
            "../inputs/sub_list2.csv", lower_bound=1000, upper_bound=1000000
        )  # time len

        random.shuffle(files)
        train_dataset = EEGDataset(
            files[1000:],
            sample_keys=["inputs", "attention_mask"],
            chunk_len=config["chunk_len"],
            num_chunks=config["num_chunks"],
            ovlp=config["chunk_ovlp"],
            root_path=root_path,
            gpt_only=not config["use_encoder"],
            normalization=config["do_normalization"],
        )

        validation_dataset = EEGDataset(
            files[:1000],
            sample_keys=["inputs", "attention_mask"],
            chunk_len=config["chunk_len"],
            num_chunks=config["num_chunks"],
            ovlp=config["chunk_ovlp"],
            root_path=root_path,
            gpt_only=not config["use_encoder"],
            normalization=config["do_normalization"],
        )

        test_dataset = None

    eval_datasets = {"validation": validation_dataset}
    if test_dataset is not None:
        eval_datasets["test"] = test_dataset

    def model_init(params: Dict = None):
        model_config = dict(config)
        if params is not None:
            model_config |= params

        return make_model(model_config)

    temp_model = model_init()
    model_summary_filepath = os.path.join(config["log_dir"], "model_summary.txt")
    with open(model_summary_filepath, "w") as f:
        f.write(str(temp_model))
        print(f"INFO: Model summary saved to {model_summary_filepath}")

    # if config["training_style"] == "decoding":
    #     model_save_steps = config["training_steps"] * 2
    # else:
    #     model_save_steps = config[
    #         "log_every_n_steps"
    #     ]

    trainer = make_trainer(
        model_init=model_init,
        training_style=config["training_style"],
        run_name=config["run_name"],
        output_dir=config["log_dir"],
        train_dataset=train_dataset,
        validation_dataset=eval_datasets,
        num_decoding_classes=config["num_decoding_classes"],
        per_device_train_batch_size=config["per_device_training_batch_size"],
        per_device_eval_batch_size=config["per_device_validation_batch_size"],
        dataloader_num_workers=config["num_workers"],
        optim=config["optim"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_beta_1"],
        adam_beta2=config["adam_beta_1"],
        adam_epsilon=config["adam_epsilon"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        max_steps=config["training_steps"],
        # num_train_epochs=5,
        logging_steps=config["log_every_n_steps"],
        eval_steps=config["eval_every_n_steps"],
        seed=(
            config["seed"] if config["set_seed"] else np.random.choice(range(1, 100000))
        ),
        fp16=config["fp16"],
        deepspeed=config["deepspeed"],
        save_steps=config[
            "eval_every_n_steps"
        ],  # necessary for "load_best_model_at_end=True"
        load_best_model_at_end=False,  # 禁用加载最佳模型，节省空间
        save_total_limit=0,  # 不保存checkpoint，只保留分析所需的小文件
        save_strategy="no",  # 完全禁用checkpoint保存
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=True,
    )

    if config["do_train"]:
        trainer.train(resume_from_checkpoint=config["resume_from"])
        # 不保存model_final以节省磁盘空间，只保留分析所需的小文件（CSV、JSON、NPY）
        # trainer.save_model(os.path.join(config["log_dir"], "model_final"))

    if test_dataset is not None:
        test_prediction = trainer.predict(test_dataset)
        pd.DataFrame(test_prediction.metrics, index=[0]).to_csv(
            os.path.join(config["log_dir"], "test_metrics.csv"), index=False
        )
        np.save(
            os.path.join(config["log_dir"], "test_predictions.npy"),
            test_prediction.predictions,
        )
        np.save(
            os.path.join(config["log_dir"], "test_label_ids.npy"),
            test_prediction.label_ids,
        )

    return trainer


def make_model(model_config: Dict = None):
    """Make model from model_config
    (as generated by get_config()).
    """

    if model_config["use_encoder"]:
        chann_coords = None

        encoder = EEGConformer(
            n_outputs=model_config["num_decoding_classes"],
            n_chans=22,
            n_times=model_config["chunk_len"],
            ch_pos=chann_coords,
            is_decoding_mode=model_config["ft_only_encoder"],
            add_log_softmax=False,  # no need softmax layer for both BCEWithLogitLoss & CrossEntropyLoss
            cls_head_layer=model_config["cls_head_layer"],  # 1ly or 3ly
        )
        # calculates the output dimension of the encoder, which is the output of transformer layer.
        model_config["parcellation_dim"] = (
            (
                model_config["chunk_len"]
                - model_config["filter_time_length"]
                + 1
                - model_config["pool_time_length"]
            )
            // model_config["stride_avg_pool"]
            + 1
        ) * model_config["n_filters_time"]

    else:
        encoder = None
        model_config["parcellation_dim"] = model_config["chunk_len"] * 22

    pos_weight_value = model_config.get("pos_weight", -1.0)
    pos_weight_tensor = None
    if model_config["training_style"] == "decoding" and pos_weight_value > 0:
        # BCEWithLogitsLoss 需要一個 Tensor，且對於二元分類，其形狀為 (1,)
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float)
        print(f"INFO: Using pos_weight = {pos_weight_value} for BCEWithLogitsLoss.")

    embedder = make_embedder(
        training_style=model_config["training_style"],
        architecture=model_config["architecture"],
        in_dim=model_config["parcellation_dim"],  # flattened, channel x chunk length
        embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_embedding_model"],
        dropout=model_config["dropout"],
        n_positions=model_config["n_positions"],
        pos_weight=pos_weight_tensor,
    )
    decoder = make_decoder(
        architecture=model_config["architecture"],
        num_hidden_layers=model_config["num_hidden_layers"],
        embed_dim=model_config["embedding_dim"],
        num_attention_heads=model_config["num_attention_heads"],
        n_positions=model_config["n_positions"],
        intermediate_dim_factor=model_config["intermediate_dim_factor"],
        hidden_activation=model_config["hidden_activation"],
        dropout=model_config["dropout"],
    )

    if model_config["embedding_dim"] != model_config["parcellation_dim"]:
        unembedder = make_unembedder(
            embed_dim=model_config["embedding_dim"],
            num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
            out_dim=model_config["parcellation_dim"],
            dropout=model_config["dropout"],
        )
    else:
        print("No Embedder and Unembedder!")
        unembedder = None

    from model import Model

    model = Model(
        encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder
    )
    if model_config["ft_only_encoder"]:
        model.switch_ft_mode(ft_encoder_only=True)

    if model_config["training_style"] == "decoding":
        model.switch_decoding_mode(
            is_decoding_mode=True,
            num_decoding_classes=model_config["num_decoding_classes"],
        )

    if model_config["pretrained_model"] is not None:
        model.from_pretrained(model_config["pretrained_model"])

    if model_config["freeze_embedder"]:
        for param in model.embedder.parameters():
            param.requires_grad = False

    if model_config["freeze_decoder"]:
        for param in model.decoder.parameters():
            param.requires_grad = False

    if model_config["freeze_encoder"]:
        for name, param in model.encoder.named_parameters():
            if "fc." in name or "final_layer" in name:
                continue
            else:
                param.requires_grad = False

    if (
        "freeze_decoder_without_pooler_heads" in model_config
        and model_config["freeze_decoder_without_pooler_heads"]
    ):
        for name, param in model.decoder.named_parameters():
            if (
                "pooler_layer" in name
                or "decoding_head" in name
                or "is_next_head" in name
            ):
                continue
            else:
                param.requires_grad = False

    if model_config["freeze_unembedder"] and unembedder is not None:
        for param in model.unembedder.parameters():
            param.requires_grad = False

    return model


def get_config(args: argparse.Namespace = None) -> Dict:
    """
    Make config from command line arguments (as created by get_args()).
    Performs additional formating of args required for calling train().
    """

    if args is None:
        args = get_args().parse_args()

    if args.smoke_test == "True":
        args.per_device_training_batch_size = 2
        args.per_device_validation_batch_size = 2
        args.training_steps = 2
        args.validation_steps = 2
        args.test_steps = 2
        args.log_every_n_steps = 1

    if args.num_attention_heads == -1:
        assert (
            args.embedding_dim % 64
        ) == 0, f"embedding-dim needs be be multiple of 64 (currently: {args.embedding_dim})"
        args.num_attention_heads = args.embedding_dim // 64

    if args.run_name == "none":
        args.run_name = f"{args.architecture}"

        if args.architecture != "LinearBaseline":

            if "Pretrained" not in args.architecture:
                args.run_name += f"_lrs-{args.num_hidden_layers}"

                args.run_name += f"_hds-{args.num_attention_heads}"

            # args.run_name += f'_embd-{args.embedding_dim}'
            # args.run_name += f'_train-{args.training_style}'
            # args.run_name += f'_lr-{str(args.learning_rate).replace(".", "")[1:]}'
            # args.run_name += f'_bs-{args.per_device_training_batch_size}'
            # args.run_name += f'_drp-{str(args.dropout).replace(".", "")}'
            args.run_name += f"_ChunkLen-{args.chunk_len}"
            args.run_name += f"_NumChunks-{args.num_chunks}"
            args.run_name += f"_ovlp-{args.chunk_ovlp}"

        else:
            args.run_name += f"_train-{args.training_style}"

        args.run_name += f"_{datetime.now().strftime('%Y-%m-%d_%H')}"

    if args.training_style == "decoding":
        args.run_name += "-" + str(args.fold_i)

    if args.smoke_test == "True":
        args.run_name = f"smoke-test_{args.run_name}"

    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.wandb_mode = (
        args.wandb_mode
        if args.wandb_mode in {"online", "offline"} and args.local_rank in {-1, 0}
        else "disabled"
    )

    config = vars(args)

    for arg in config:

        if config[arg] in {"True", "False"}:
            config[arg] = config[arg] == "True"

        elif config[arg] == "none":
            config[arg] = None

        elif "subjects_per_dataset" in arg:
            config[arg] = None if config[arg] == -1 else config[arg]

    return config


def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(description="run model training")

    parser.add_argument(
        "--dataset-name",
        metavar="STR",
        default="bci2a",
        choices=("KaggleERN", "stress", "motor6class", "emotion7class"),
        type=str,
        help="supported downstream dataset",
    )
    # Data pipeline settings:
    parser.add_argument(
        "--train-data-path",
        metavar="DIR",
        default="../../tuh_tensors/",
        type=str,
        help="path to training data directory " "(default: data/upstream)",
    )

    parser.add_argument(
        "--dst-data-path",
        metavar="DIR",
        default="../../bci2a_egg_npz/",
        type=str,
        help="path to training data directory " "(default: data/upstream)",
    )

    parser.add_argument(
        "--matrix_p_path",
        metavar="DIR",
        default="../../bci2a_egg_npz/",
        type=str,
        help="path to P matrix of specific downstream task",
    )

    parser.add_argument(
        "--cls_head_layer",
        metavar="STR",
        default="3ly",
        choices=("1ly", "3ly"),
        type=str,
        help="number of layers",
    )
    parser.add_argument(
        "--parcellation-dim",
        metavar="INT",
        default=1024,
        type=int,
        help="dimension of input data parcellation (default: 1024). "
        "! This is fixed for the current up-/downstream data.",
    )
    parser.add_argument(
        "--pretrained-model",
        metavar="DIR",
        type=str,
        default="none",
        help="checkpoint used to initialize model weights " "(default: none)",
    )

    # Embedder settings:
    parser.add_argument(
        "--embedding-dim",
        metavar="INT",
        default=1024,
        type=int,
        help="dimension of input embedding " "(default: 1024)",
    )
    parser.add_argument(
        "--num-hidden-layers-embedding-model",
        metavar="INT",
        default=1,
        type=int,
        help="numer of layers of linear embedding model " "(default: 1)",
    )
    parser.add_argument(
        "--freeze-embedder",
        metavar="BOOL",
        default="False",
        choices=("True", "False"),
        type=str,
        help="whether or not to freeze embedder weights during training "
        "(default: False) ",
    )

    # UnEmbedder settings:
    parser.add_argument(
        "--num-hidden-layers-unembedding-model",
        metavar="INT",
        default=1,
        type=int,
        help="numer of hidden layers for linear unembedding model " "(default: 1)",
    )
    parser.add_argument(
        "--freeze-unembedder",
        metavar="BOOL",
        default="False",
        choices=("True", "False"),
        type=str,
        help="whether or not to freeze unembedder weights during training "
        "(default: False) ",
    )

    # Decoder settings:
    parser.add_argument(
        "--architecture",
        metavar="STR",
        default="GPT",
        choices=("GPT", "PretrainedGPT2"),
        type=str,
        help="Model architecture used for sequence modeling / decoding. "
        "(default: GPT) ",
    )
    parser.add_argument(
        "--num-hidden-layers",
        metavar="INT",
        default=4,
        type=int,
        help="number of hidden model layers in --architecture "
        "(default: 4). "
        "! Does not apply to LinearBaseline; "
        "! Same number of hidden layers is used for decoder / encoder "
        "parts of autoencoder (ie., default creates encoder and decoder "
        "with 4 hidden layers each)",
    )
    parser.add_argument(
        "--num-attention-heads",
        metavar="INT",
        default=-1,
        type=int,
        help="number of attention heads per transformer layer "
        "(default: embedding-dim // 64). "
        "! Does not apply to non-transformer models",
    )
    parser.add_argument(
        "--intermediate-dim-factor",
        metavar="INT",
        default=4,
        type=int,
        help="scales feed-forward transformer layer dimension relative to "
        "embedding-dim: intermediate-dim-factor * embedding-dim "
        "(default: 4)",
    )
    parser.add_argument(
        "--hidden-activation",
        metavar="STR",
        default="gelu_new",
        choices=("gelu", "gelu_new", "relu", "silu"),
        type=str,
        help="type of hidden activation of transformer layers "
        "(default: gelu_new); "
        'one of {"gelu", "gelu_new", "relu", "silu"}. '
        "! Does not apply to non-transformer models",
    )
    parser.add_argument(
        "--freeze-decoder",
        metavar="BOOL",
        default="False",
        choices=("True", "False"),
        type=str,
        help="whether or not to freeze decoder model weights during training "
        "as specified by --architecture "
        "(default: False) ",
    )
    parser.add_argument(
        "--freeze-decoder-without-pooler-heads",
        metavar="BOOL",
        default="False",
        choices=("True", "False"),
        type=str,
        help="whether or not to freeze decoder model weights during training "
        "as specified by --architecture, without pooler layer and "
        " is-next-pred / decoding heads "
        "(default: False) ",
    )

    # Classifier setting:
    parser.add_argument(
        "--pos-weight",
        metavar="FLOAT",
        default=-1.0,
        type=float,
        help="Positive class weight for BCEWithLogitsLoss to handle class imbalance. "
        "Typically set to (num_neg / num_pos). "
        "If set to -1, no weight is used (default: -1).",
    )

    # Trainer settings:
    parser.add_argument(
        "--resume-from",
        metavar="DIR",
        type=str,
        default="none",
        help="continue training from specified checkpoint " "(default: none)",
    )
    parser.add_argument(
        "--training-style",
        metavar="STR",
        default="CSM_causal",
        choices=("CSM", "CSM_causal", "decoding"),
        type=str,
        help="training framework / style (default: CSM); " "one of CSM, decoding",
    )
    parser.add_argument(
        "--decoding-target",
        metavar="STR",
        default="none",
        type=str,
        help="key for decoding target variable in .tar-files in --data"
        "(default: none). "
        '! Must be specified when setting --training-style to "decoding"',
    )
    parser.add_argument(
        "--num-decoding-classes",
        metavar="INT",
        default=4,
        type=int,
        help="number of decoding classes (ie., mental states) in --data "
        "(default: 0). "
        '! Must be specified when setting --training-style to "decoding"',
    )
    parser.add_argument(
        "--training-steps",
        metavar="INT",
        default=60000,
        type=int,
        help="number of training steps to perform " "(default: 400000)",
    )
    parser.add_argument(
        "--validation-steps",
        metavar="INT",
        default=1000,
        type=int,
        help="number of validation steps to perform at evaluation time "
        "(default: 1000)",
    )
    parser.add_argument(
        "--test-steps",
        metavar="INT",
        default=1000,
        type=int,
        help="number of test steps to perform at test time"
        "(default: 2000). "
        "! Test evaluation only performed if test set created by "
        "setting --n-test-subjects-per-dataset != -1",
    )
    parser.add_argument(
        "--per-device-training-batch-size",
        metavar="INT",
        default=16,
        type=int,
        help="batch size during training per training device " "(default: 64)",
    )
    parser.add_argument(
        "--per-device-validation-batch-size",
        metavar="INT",
        default=16,
        type=int,
        help="batch size during evaluation per training device " "(default: 64)",
    )
    parser.add_argument(
        "--optim",
        metavar="STR",
        default="adamw_hf",
        type=str,
        help="optimizer to use for training "
        "(default: adamw_hf) -> adamw from HuggingFrace transformer library. "
        "For other options see Huggingface TrainerArgs.",
    )
    parser.add_argument(
        "--learning-rate",
        metavar="FLOAT",
        default=1e-4,
        type=float,
        help="maximum learning rate during training " "(default: 1e-4)",
    )
    parser.add_argument(
        "--warmup-ratio",
        metavar="FLOAT",
        default=0.01,
        type=float,
        help="warm-up steps for linear learning rate scheduler "
        "specified as fraction of --training-steps "
        "(default: 0.01)",
    )
    parser.add_argument(
        "--weight-decay",
        metavar="FLOAT",
        default=0.1,
        type=float,
        help="weight decay strength (indicating l2-regularisation strength) "
        "(default: 0.1)",
    )
    parser.add_argument(
        "--adam-beta-1",
        metavar="FLOAT",
        default=0.9,
        type=float,
        help="adam beta 1 (default: 0.9)",
    )
    parser.add_argument(
        "--adam-beta-2",
        metavar="FLOAT",
        default=0.999,
        type=float,
        help="adam beta 2 (default: 0.999)",
    )
    parser.add_argument(
        "--adam-epsilon",
        metavar="FLOAT",
        default=1e-8,
        type=float,
        help="adam beta 2 (default: 1e-8)",
    )
    parser.add_argument(
        "--max-grad-norm",
        metavar="FLOAT",
        default=1.0,
        type=float,
        help="maximum gradient clipping norm (default: 1.0)",
    )
    parser.add_argument(
        "--lr-scheduler",
        metavar="STR",
        default="linear",
        choices=("linear", "constant_with_warmup", "none"),
        type=str,
        help="learning rate scheduler; "
        "one of {linear, constant_with_warmup, none} "
        "(default: linear)",
    )
    parser.add_argument(
        "--dropout",
        metavar="FLOAT",
        default=0.1,
        type=float,
        help="dropout ratio for hidden layers of embedder and decoder model parts "
        "(default: 0.1)",
    )
    parser.add_argument(
        "--pos_weight",
        metavar="FLOAT",
        default=0.413,
        type=float,
        help="dropout ratio for hidden layers of embedder and decoder model parts "
        "(default: 0.1)",
    )

    # Logging settings:
    parser.add_argument(
        "--log-dir",
        metavar="DIR",
        type=str,
        default="results/models/upstream",
        help="path where training is logged " "(default: results/models/upstream)",
    )
    parser.add_argument(
        "--log-every-n-steps",
        metavar="INT",
        default=1000,
        type=int,
        help="frequence of logging in training steps " "(default: 10000)",
    )
    parser.add_argument(
        "--run-name",
        metavar="STR",
        type=str,
        default="none",
        help="descriptor of the training run used for logging and wandb; "
        '! if set to "none", a unique identifier is automatically created',
    )
    parser.add_argument(
        "--wandb-mode",
        metavar="STR",
        choices=("online", "offline", "disabled"),
        default="disabled",
        help="track training w/ wandb online or offline or not at all "
        "(default: disabled) "
        "! requires setting up weights-and-bias for this machine; "
        "see: https://docs.wandb.ai/",
    )
    parser.add_argument(
        "--wandb-project-name",
        metavar="STR",
        type=str,
        default="learning-from-brains",
        help="name of wandb project where data is logged "
        "(default: learning-from-brains)",
    )

    # Other settings:
    parser.add_argument(
        "--seed",
        metavar="INT",
        default=1234,
        type=int,
        help="random seed (default: 1234)",
    )
    parser.add_argument(
        "--set-seed",
        metavar="BOOL",
        choices=("True", "False"),
        default="True",
        type=str,
        help="whether or not to set random seed (default: True)",
    )
    parser.add_argument(
        "--fp16",
        metavar="BOOL",
        choices=("True", "False"),
        default="True",
        help="whether or not to use 16-bit precision GPU training " "(default: True)",
    )
    parser.add_argument(
        "--deepspeed",
        metavar="DIR",
        default="none",
        type=str,
        help="location of deepspeed configuration file; "
        "automatically adds deepspeed functionality to training if specified "
        "(default: none)",
    )
    parser.add_argument(
        "--local_rank",
        metavar="INT",
        default=-1,
        type=int,
        help="Rank of the process during distributed training " "(default: -1)",
    )
    parser.add_argument(
        "--num-workers",
        metavar="INT",
        default=8,
        type=int,
        help="number of data loading workers " "(default: 0 -> load in main process)",
    )
    parser.add_argument(
        "--plot-model-graph",
        metavar="BOOL",
        default="False",
        type=str,
        choices=("True", "False"),
        help="whether or not to save an image of the model graph to log-dir "
        "(default: False)",
    )
    parser.add_argument(
        "--smoke-test",
        metavar="BOOL",
        default="False",
        type=str,
        choices=("True", "False"),
        help="whetehr or not to run training in smoke test-mode "
        "(default: False)"
        'If set to "True", training is restricted by setting: '
        "--per-device-training_batch_size 2 "
        "--per-device-validation_batch_size 2 "
        "--training-steps 2 "
        "--validation-steps 2 "
        "--test-steps 2 "
        "--log-every-n-steps 1",
    )
    parser.add_argument(
        "--bold-dummy-mode",
        metavar="BOOL",
        default="False",
        type=str,
        choices=("True", "False"),
        help="whether or not to replace BOLD with dummy during training; "
        "for internal testing purposes only! "
        "(default: False)",
    )
    parser.add_argument(
        "--do-train",
        metavar="BOOL",
        default="True",
        type=str,
        choices=("True", "False"),
        help="whether or not to run training "
        "(default: True). "
        'If "False", train() still returns trainer',
    )

    parser.add_argument(
        "--n-positions",
        metavar="INT",
        default=512,
        type=int,
        help="maximum sequence length that transformer model might ever be used with "
        "(default: 512)",
    )
    ## EEG settings
    parser.add_argument("--chunk_len", default=500, type=int)
    parser.add_argument("--num_chunks", default=8, type=int)
    parser.add_argument("--chunk_ovlp", default=50, type=int)
    parser.add_argument("--sampling_rate", default=250, type=int)
    parser.add_argument("--fold_i", default=0, type=int)

    parser.add_argument(
        "--use-encoder",
        metavar="BOOL",
        default="True",
        type=str,
        choices=("True", "False"),
        help="whether to use encoder or not",
    )
    parser.add_argument(
        "--do-normalization",
        metavar="BOOL",
        default="True",
        type=str,
        choices=("True", "False"),
        help="whether to use encoder or not",
    )

    parser.add_argument(
        "--filter-time-length",
        metavar="INT",
        default=25,
        type=int,
        help="length of the temporal filter (default: 25)",
    )
    parser.add_argument(
        "--pool-time-length",
        metavar="INT",
        default=75,
        type=int,
        help="length of temporal pooling filter (default: 75)",
    )
    parser.add_argument(
        "--stride-avg-pool",
        metavar="INT",
        default=15,
        type=int,
        help="length of stride between temporal pooling filters (default: 15)",
    )
    parser.add_argument(
        "--n-filters-time",
        metavar="INT",
        default=40,
        type=int,
        help="number of temporal filters (default: 40)",
    )
    parser.add_argument(
        "--num-encoder-layers",
        metavar="INT",
        default=6,
        type=int,
        help="number of transformer layers in encoder",
    )

    parser.add_argument("--eval_every_n_steps", default=200, type=int)
    parser.add_argument(
        "--freeze-encoder",
        metavar="BOOL",
        default="False",
        choices=("True", "False"),
        type=str,
        help="whether or not to freeze encoder weights during training "
        "(default: False) ",
    )
    parser.add_argument(
        "--ft-only-encoder",
        metavar="BOOL",
        default="False",
        choices=("True", "False"),
        type=str,
        help="finetune with only encoder or not " "(default: False) ",
    )
    parser.add_argument(
        "--metric_for_best_model",
        metavar="STR",
        default="eval_validation_loss",
        choices=(
            "eval_validation_loss", 
            "eval_validation_acc", 
            "eval_validation_accuracy",
            "eval_validation_bacc",
            "eval_validation_kappa",
            "eval_validation_f1_weighted",
            "eval_validation_roc_auc_weighted",
        ),
        type=str,
        help="record metric for saving trained model",
    )

    return parser


if __name__ == "__main__":

    trainer = train()
