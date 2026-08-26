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
from batcher.downstream_dataset import KaggleERNDataset, Motor6ClassDataset, Emotion7ClassDataset, Sleep5ClassDataset
import torch
import sys
import os
import glob
import argparse
import pdb
import time
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

from utils import read_threshold_sub, list_motor_files_by_subject, list_stress_files_by_subject

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, "../"))
# from batcher.make import make_batcher
from batcher.base import EEGDataset
from decoder.make_decoder import make_decoder
from embedder.make import make_embedder
from trainer.make import make_trainer, compute_voting_metrics, extract_video_index, extract_subject_id
from trainer.base import Trainer
from decoder.unembedder import make_unembedder

os.environ["WANDB_DISABLED"] = "true"


def prepare_motor6class_dataset_subject_independent(
    downstream_path, test_subject, val_subject, matrix_p_path, chunk_len, num_chunks, ovlp, gpt_only,
    normalization=True,
):
    """[LOSO] 20折 subject-independent 划分：test = test_subject, val = val_subject,
    train = 其余全部受试者。与 cbramod_finetune/datasets/motortask_dataset.py、
    biot_finetune/eegpt_finetune 的 Motion 划分、labram_finetune 的
    prepare_Motor_dataset_subject_independent 使用同一批底层 AllSubjects_Epochs
    pickle 文件、同一套受试者提取规则，保证同一折在不同模型间的
    train/val/test 受试者集合完全一致。
    """
    subject_to_files = list_motor_files_by_subject(downstream_path)
    subjects = sorted(subject_to_files.keys(), key=lambda s: int(s[3:]))
    if len(subjects) < 3:
        raise ValueError(f"Need at least 3 subjects for subject-independent split, got {len(subjects)}: {subjects}")
    for s in (test_subject, val_subject):
        if s not in subject_to_files:
            raise ValueError(f"Subject {s!r} not found among {subjects}")
    if test_subject == val_subject:
        raise ValueError("test_subject and val_subject must be different")

    train_subjects = [s for s in subjects if s not in (test_subject, val_subject)]

    def gather(subj_list):
        files = []
        for s in subj_list:
            files.extend(subject_to_files[s])
        return files

    train_files = gather(train_subjects)
    val_files = gather([val_subject])
    test_files = gather([test_subject])

    print("=" * 70)
    print(f"[split_mode=subject_independent] test={test_subject} val={val_subject} "
          f"train={len(train_subjects)} subjects")
    print(f"  All subjects ({len(subjects)}): {subjects}")
    print(f"  file counts: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print("=" * 70)

    # list_motor_files_by_subject 已经返回绝对路径；root_path 传 downstream_path
    # 只是为了满足 EEGDataset.__init__ 里 root_path != "" 的分支，绝对路径本身
    # 会让 os.path.join(root_path, abs_path) 直接忽略 root_path（no-op）。
    train_dataset = Motor6ClassDataset(
        train_files, sample_keys=["inputs", "attention_mask"], chunk_len=chunk_len, num_chunks=num_chunks,
        ovlp=ovlp, root_path=downstream_path, matrix_p_path=matrix_p_path, gpt_only=gpt_only,
        normalization=normalization,
    )
    validation_dataset = Motor6ClassDataset(
        val_files, sample_keys=["inputs", "attention_mask"], chunk_len=chunk_len, num_chunks=num_chunks,
        ovlp=ovlp, root_path=downstream_path, matrix_p_path=matrix_p_path, gpt_only=gpt_only,
        normalization=normalization,
    )
    # test_dataset 需要 sample_id/subject_id 才能事后从保存的 npz 重新算所有指标。
    test_dataset = Motor6ClassDataset(
        test_files, sample_keys=["inputs", "attention_mask"], chunk_len=chunk_len, num_chunks=num_chunks,
        ovlp=ovlp, root_path=downstream_path, matrix_p_path=matrix_p_path, gpt_only=gpt_only,
        return_sample_id=True, normalization=normalization,
    )
    return train_dataset, validation_dataset, test_dataset


def prepare_stress_dataset_subject_independent(
    downstream_path, test_subject, val_subject, matrix_p_path, chunk_len, num_chunks, ovlp, gpt_only,
):
    """[LOSO] Stress 任务的 subject-independent 划分：test = test_subject, val = val_subject,
    train = 其余全部受试者。与 cbramod_finetune/datasets/custom_stress_dataset.py、
    labram_finetune/neurolm_finetune/eegpt_finetune 的 Stress LOSO 划分使用同一批底层
    stress_data 预处理生成的 pickle 文件（'SubNN_{increase,normal}_edfNN_chunkNNNN.pickle'）、
    同一套受试者提取规则，保证同一折在不同模型间的 train/val/test 受试者集合完全一致。
    """
    subject_to_files = list_stress_files_by_subject(downstream_path)
    subjects = sorted(subject_to_files.keys(), key=lambda s: int(s[3:]))
    if len(subjects) < 3:
        raise ValueError(f"Need at least 3 subjects for subject-independent split, got {len(subjects)}: {subjects}")
    for s in (test_subject, val_subject):
        if s not in subject_to_files:
            raise ValueError(f"Subject {s!r} not found among {subjects}")
    if test_subject == val_subject:
        raise ValueError("test_subject and val_subject must be different")

    train_subjects = [s for s in subjects if s not in (test_subject, val_subject)]

    def gather(subj_list):
        files = []
        for s in subj_list:
            files.extend(subject_to_files[s])
        return files

    train_files = gather(train_subjects)
    val_files = gather([val_subject])
    test_files = gather([test_subject])

    print("=" * 70)
    print(f"[split_mode=subject_independent] test={test_subject} val={val_subject} "
          f"train={len(train_subjects)} subjects")
    print(f"  All subjects ({len(subjects)}): {subjects}")
    print(f"  file counts: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print("=" * 70)

    # list_stress_files_by_subject 已经返回绝对路径；root_path 传 downstream_path
    # 只是为了满足 EEGDataset.__init__ 里 root_path != "" 的分支，绝对路径本身
    # 会让 os.path.join(root_path, abs_path) 直接忽略 root_path（no-op）。
    train_dataset = KaggleERNDataset(
        train_files, sample_keys=["inputs", "attention_mask"], chunk_len=chunk_len, num_chunks=num_chunks,
        ovlp=ovlp, root_path=downstream_path, matrix_p_path=matrix_p_path, gpt_only=gpt_only,
    )
    validation_dataset = KaggleERNDataset(
        val_files, sample_keys=["inputs", "attention_mask"], chunk_len=chunk_len, num_chunks=num_chunks,
        ovlp=ovlp, root_path=downstream_path, matrix_p_path=matrix_p_path, gpt_only=gpt_only,
    )
    # test_dataset 需要 sample_id/subject_id 才能事后从保存的 npz 重新算所有指标。
    test_dataset = KaggleERNDataset(
        test_files, sample_keys=["inputs", "attention_mask"], chunk_len=chunk_len, num_chunks=num_chunks,
        ovlp=ovlp, root_path=downstream_path, matrix_p_path=matrix_p_path, gpt_only=gpt_only,
        return_sample_id=True,
    )
    return train_dataset, validation_dataset, test_dataset


def save_loso_fold_results(config: Dict, trainer: Trainer, test_dataset, test_prediction, train_time_sec: float) -> None:
    """[LOSO] 保存单折 Motor6Class LOSO 结果：
      {task}_{model}_fold{i:02d}.npz -- sample_id / y_true / y_pred / y_prob(softmax) / subject_id
      {task}_{model}_fold{i:02d}.json -- fold / test_subject / val_subject / balanced_accuracy /
                                          best_epoch / hyperparams / train_time_sec /
                                          peak_gpu_mem_mb / gpu_name
    直接复用 trainer.predict(test_dataset) 已经算出来的结果（test_dataset 以
    return_sample_id=True 构造，predict() 内部走的是确定性的 SequentialSampler，
    顺序与 test_dataset.sample_ids/.subject_ids 逐条对应），不重新加载模型/
    重新推理，也不改动训练逻辑。best_epoch 取自 trainer.state（HF Trainer 的
    load_best_model_at_end 已经按 --metric_for_best_model 选出了最佳checkpoint，
    这里只是把同一个选择记录下来）。任何失败都直接抛异常退出，不静默跳过。
    """
    if not getattr(test_dataset, "return_sample_id", False):
        raise RuntimeError("save_loso_fold_results requires test_dataset built with return_sample_id=True")

    sample_ids = test_dataset.sample_ids
    subject_ids = test_dataset.subject_ids
    n = len(sample_ids)
    if n == 0:
        raise RuntimeError("save_loso_fold_results: test_dataset has 0 samples, refusing to save an empty file.")

    preds_logits = np.asarray(test_prediction.predictions)
    y_true_raw = np.asarray(test_prediction.label_ids)
    if len(preds_logits) != n or len(y_true_raw) != n:
        raise RuntimeError(
            f"save_loso_fold_results: prediction length mismatch "
            f"(predictions={len(preds_logits)}, labels={len(y_true_raw)}, expected {n} from test_dataset)"
        )

    # encoder/base.py's EEGModuleMixin.__init__ collapses n_outputs<=2 to a single
    # logit ("Use BCEWithLogitLoss for Binary calssification") whenever
    # --ft-only-encoder=True (as this LOSO launcher always uses) -- so a binary task
    # like Stress (--num-decoding-classes=2) actually gets a (N,1)-shaped single-logit
    # sigmoid output here, not (N,2) softmax logits. Blindly doing softmax(dim=-1) on a
    # (N,1) array is a no-op (always 1.0) and argmax(axis=-1) is always 0 regardless of
    # the true label -- detect this the same way make_decoding_accuracy_metrics() in
    # trainer/make.py already does (shape[-1]==1), and branch to sigmoid+threshold,
    # matching the (N,2) [P(class0), P(class1)] npz schema every sibling model
    # (cbramod/labram/neurolm's single-logit heads) already uses so
    # compute_metrics_from_npz.py's y_prob.argmax(axis=1)==y_pred check still holds.
    is_binary_logit_output = preds_logits.shape[-1] == 1 or preds_logits.ndim == 1
    if is_binary_logit_output:
        logits_1d = preds_logits.reshape(-1).astype(np.float32)
        prob_pos = torch.sigmoid(torch.from_numpy(logits_1d)).numpy()
        y_pred_all = (logits_1d > 0).astype(np.int64)
        y_prob_all = np.stack([1.0 - prob_pos, prob_pos], axis=-1)
    else:
        y_prob_all = torch.nn.functional.softmax(torch.from_numpy(preds_logits).float(), dim=-1).numpy()
        y_pred_all = preds_logits.argmax(axis=-1)

    sample_ids_arr = np.array(sample_ids)
    order = np.argsort(sample_ids_arr)
    sample_ids_arr = sample_ids_arr[order]
    y_true_arr = y_true_raw.astype(np.int64)[order]
    y_pred_arr = y_pred_all.astype(np.int64)[order]
    y_prob_arr = y_prob_all.astype(np.float32)[order]
    subject_id_arr = np.array(subject_ids, dtype=np.int64)[order]

    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    # best_epoch/best_step: HF Trainer 的 state.log_history 里，验证集
    # --metric_for_best_model 达到 state.best_metric 的那条记录对应的 epoch/step。
    metric_key = config["metric_for_best_model"]
    best_epoch, best_step = None, None
    if trainer.state.best_metric is not None:
        for entry in trainer.state.log_history:
            if metric_key in entry and abs(entry[metric_key] - trainer.state.best_metric) < 1e-6:
                best_epoch = entry.get("epoch")
                best_step = entry.get("step")
                break
    if best_epoch is None:
        raise RuntimeError(
            "save_loso_fold_results: could not determine best_epoch from trainer.state "
            f"(best_metric={trainer.state.best_metric}, metric_for_best_model={metric_key!r})."
        )

    task = config.get("task_name") or config["dataset_name"]
    model_name = config.get("model_name") or "neurogpt"
    fold_idx = config["fold_idx"]
    save_dir = config.get("fold_results_dir") or config["log_dir"]
    if not save_dir:
        raise ValueError("save_loso_fold_results requires --fold-results-dir or --log-dir to be set")
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(save_dir, f"{task}_{model_name}_fold{fold_idx:02d}.npz")
    json_path = os.path.join(save_dir, f"{task}_{model_name}_fold{fold_idx:02d}.json")

    # 已有旧结果先改名备份，绝不静默覆盖
    for path in (npz_path, json_path):
        if os.path.exists(path):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = f"{path}.bak-{ts}"
            os.rename(path, backup_path)
            print(f"[warn] existing fold result found, backed up to {backup_path}")

    np.savez(
        npz_path,
        sample_id=sample_ids_arr,
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_prob=y_prob_arr,
        subject_id=subject_id_arr,
    )
    if not os.path.exists(npz_path):
        raise RuntimeError(f"save_loso_fold_results: failed to write {npz_path}")
    _reload = np.load(npz_path)
    for key in ("sample_id", "y_true", "y_pred", "y_prob", "subject_id"):
        if key not in _reload:
            raise RuntimeError(f"save_loso_fold_results: {npz_path} missing key {key!r} after write")
        if len(_reload[key]) != n:
            raise RuntimeError(f"save_loso_fold_results: {npz_path} key {key!r} length mismatch after write")

    hyperparams = {
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "per_device_training_batch_size": config["per_device_training_batch_size"],
        "training_steps": config["training_steps"],
        "eval_every_n_steps": config["eval_every_n_steps"],
        "optim": config["optim"],
        "seed": config["seed"],
        "metric_for_best_model": metric_key,
        "cls_head_layer": config["cls_head_layer"],
        "ft_only_encoder": config["ft_only_encoder"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_encoder_layers": config["num_encoder_layers"],
        "embedding_dim": config["embedding_dim"],
    }

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated(0) / (1024 ** 2)) if torch.cuda.is_available() else None

    meta = {
        "fold": fold_idx,
        "test_subject": config.get("test_subject"),
        "val_subject": config.get("val_subject"),
        "balanced_accuracy": balanced_accuracy,
        "best_epoch": int(round(best_epoch)),
        "best_epoch_exact": best_epoch,
        "best_step": best_step,
        "saved_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "hyperparams": hyperparams,
        "train_time_sec": train_time_sec,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "gpu_name": gpu_name,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if not os.path.exists(json_path):
        raise RuntimeError(f"save_loso_fold_results: failed to write {json_path}")
    with open(json_path) as f:
        json.load(f)  # 回读校验 JSON 没写坏

    print(f"Saved fold predictions to {npz_path}")
    print(f"Saved fold metadata to {json_path}")
    print(f"  fold={fold_idx} test={config.get('test_subject')} val={config.get('val_subject')} "
          f"best_epoch={meta['best_epoch']} balanced_accuracy={balanced_accuracy:.5f}")


def _sleep_prediction_to_npz_fields(dataset, prediction):
    """[Sleep] 把 trainer.predict(dataset) 的结果（dataset 以 return_sample_id=True
    构造，predict() 内部走确定性的 SequentialSampler，顺序与 dataset.sample_ids/
    .subject_ids/.epoch_indices 逐条对应）转成排好序的 sample_id/y_true/y_pred/
    y_prob/subject_id/epoch_index 数组。跟 save_loso_fold_results 的二分类/多分类
    判定逻辑一致（sleep5class 是 5 分类，这里始终走 softmax 分支，但保留同样的
    (N,1) 兜底判断以防 num-decoding-classes 传错）。
    """
    sample_ids = dataset.sample_ids
    subject_ids = dataset.subject_ids
    epoch_indices = dataset.epoch_indices
    n = len(sample_ids)
    if n == 0:
        raise RuntimeError("_sleep_prediction_to_npz_fields: dataset has 0 samples, refusing to save an empty file.")

    preds_logits = np.asarray(prediction.predictions)
    y_true_raw = np.asarray(prediction.label_ids)
    if len(y_true_raw) != n:
        raise RuntimeError(
            f"_sleep_prediction_to_npz_fields: label length mismatch "
            f"(labels={len(y_true_raw)}, expected {n} from dataset)"
        )

    # "decoding" 训练模式下模型对每个样本输出的是逐 chunk(时间步)的 logits，不是
    # 一个样本一个 logit 向量——preds_logits 的长度会是 n * num_chunks_effective
    # （sleep5class 实测 num_chunks=30 时是 n*15，说明 encoder 内部把 30 个 chunk
    # 降采样成了 15 个位置），跟 label_ids 的长度 n 对不上。这跟
    # trainer/make.py::make_decoding_accuracy_metrics()（--metric_for_best_model
    # 用的就是它，决定了 load_best_model_at_end 选哪个 checkpoint）处理的是同一个
    # 情况，这里复刻它的做法：reshape 成 (n, num_chunks_effective, num_classes) 后
    # 对 chunk 维度取平均 logits，再 softmax/argmax——只有这样这里重新算出来的
    # y_pred/balanced_accuracy 才会跟训练时汇报的 eval_validation_bacc 对得上。
    if len(preds_logits) != n:
        if len(preds_logits) % n != 0:
            raise RuntimeError(
                f"_sleep_prediction_to_npz_fields: prediction length {len(preds_logits)} is not a "
                f"multiple of dataset length {n}; cannot reshape into (n, num_chunks, num_classes)."
            )
        num_chunks_effective = len(preds_logits) // n
        preds_logits = preds_logits.reshape(n, num_chunks_effective, -1).mean(axis=1)
        print(f"[info] pooled decoding-style predictions: {n}*{num_chunks_effective} chunks -> {n} epoch-level logits")

    if preds_logits.shape[-1] == 1 or preds_logits.ndim == 1:
        logits_1d = preds_logits.reshape(-1).astype(np.float32)
        prob_pos = torch.sigmoid(torch.from_numpy(logits_1d)).numpy()
        y_pred_all = (logits_1d > 0).astype(np.int64)
        y_prob_all = np.stack([1.0 - prob_pos, prob_pos], axis=-1)
    else:
        y_prob_all = torch.nn.functional.softmax(torch.from_numpy(preds_logits).float(), dim=-1).numpy()
        y_pred_all = preds_logits.argmax(axis=-1)

    sample_ids_arr = np.array(sample_ids)
    order = np.argsort(sample_ids_arr)
    return {
        "sample_id": sample_ids_arr[order],
        "y_true": y_true_raw.astype(np.int64)[order],
        "y_pred": y_pred_all.astype(np.int64)[order],
        "y_prob": y_prob_all.astype(np.float32)[order],
        "subject_id": np.array(subject_ids, dtype=np.int64)[order],
        "epoch_index": np.array(epoch_indices, dtype=np.int64)[order],
    }


def save_sleep_epoch_results(config: Dict, trainer: Trainer, val_dataset, test_dataset,
                              val_prediction, test_prediction, train_time_sec: float) -> None:
    """[Sleep] 保存 sleep5class 的 val/test 结果：
      sleep_{model}_val.npz / sleep_{model}_test.npz
        -- sample_id(=epoch_id，已排序) / y_true / y_pred / y_prob(N,5) / subject_id / epoch_index
      sleep_{model}.json
        -- 模型名/任务/lr/wd/bs/best_epoch/val_bacc/test_bacc/数据目录/val与test各类样本数
    跟 save_loso_fold_results 同一个思路：直接复用 trainer.predict() 已经算出来的
    结果（val_dataset/test_dataset 都以 return_sample_id=True 构造），不重新加载
    模型/重新推理，也不改动训练逻辑。Sleep 不是 LOSO，没有 fold/test_subject/
    val_subject，所以单独写一份，不复用 save_loso_fold_results 的文件名/字段。
    任何失败都直接抛异常退出，不静默跳过。
    """
    if not (getattr(val_dataset, "return_sample_id", False) and getattr(test_dataset, "return_sample_id", False)):
        raise RuntimeError("save_sleep_epoch_results requires val/test datasets built with return_sample_id=True")

    val_data = _sleep_prediction_to_npz_fields(val_dataset, val_prediction)
    test_data = _sleep_prediction_to_npz_fields(test_dataset, test_prediction)

    task = config.get("task_name") or config["dataset_name"]
    model_name = config.get("model_name") or "neurogpt"
    save_dir = config.get("fold_results_dir") or config["log_dir"]
    if not save_dir:
        raise ValueError("save_sleep_epoch_results requires --fold-results-dir or --log-dir to be set")
    os.makedirs(save_dir, exist_ok=True)

    val_npz_path = os.path.join(save_dir, f"{task}_{model_name}_val.npz")
    test_npz_path = os.path.join(save_dir, f"{task}_{model_name}_test.npz")
    for path, data in ((val_npz_path, val_data), (test_npz_path, test_data)):
        if os.path.exists(path):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = f"{path}.bak-{ts}"
            os.rename(path, backup_path)
            print(f"[warn] existing sleep result found, backed up to {backup_path}")
        np.savez(path, **data)
        if not os.path.exists(path):
            raise RuntimeError(f"save_sleep_epoch_results: failed to write {path}")
        _reload = np.load(path)
        for key in ("sample_id", "y_true", "y_pred", "y_prob", "subject_id", "epoch_index"):
            if key not in _reload:
                raise RuntimeError(f"save_sleep_epoch_results: {path} missing key {key!r} after write")
            if len(_reload[key]) != len(data["sample_id"]):
                raise RuntimeError(f"save_sleep_epoch_results: {path} key {key!r} length mismatch after write")

    from sklearn.metrics import balanced_accuracy_score
    val_bacc = float(balanced_accuracy_score(val_data["y_true"], val_data["y_pred"]))
    test_bacc = float(balanced_accuracy_score(test_data["y_true"], test_data["y_pred"]))
    n_classes = config.get("num_decoding_classes") or 5
    val_class_counts = np.bincount(val_data["y_true"], minlength=n_classes).tolist()
    test_class_counts = np.bincount(test_data["y_true"], minlength=n_classes).tolist()

    metric_key = config["metric_for_best_model"]
    best_epoch, best_step = None, None
    if trainer.state.best_metric is not None:
        for entry in trainer.state.log_history:
            if metric_key in entry and abs(entry[metric_key] - trainer.state.best_metric) < 1e-6:
                best_epoch = entry.get("epoch")
                best_step = entry.get("step")
                break
    if best_epoch is None:
        raise RuntimeError(
            "save_sleep_epoch_results: could not determine best_epoch from trainer.state "
            f"(best_metric={trainer.state.best_metric}, metric_for_best_model={metric_key!r})."
        )

    hyperparams = {
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "per_device_training_batch_size": config["per_device_training_batch_size"],
        "training_steps": config["training_steps"],
        "eval_every_n_steps": config["eval_every_n_steps"],
        "optim": config["optim"],
        "seed": config["seed"],
        "metric_for_best_model": metric_key,
        "cls_head_layer": config["cls_head_layer"],
        "ft_only_encoder": config["ft_only_encoder"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_encoder_layers": config["num_encoder_layers"],
        "embedding_dim": config["embedding_dim"],
    }

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated(0) / (1024 ** 2)) if torch.cuda.is_available() else None

    meta = {
        "model_name": model_name,
        "task": task,
        "dataset": config["dataset_name"],
        "split_mode": "pooled_random_epoch",  # 见 preprocess_sleep.py：全体受试者按 epoch 分层随机切分，不是 LOSO
        "dataset_path": config.get("dst_data_path"),
        "best_epoch": int(round(best_epoch)),
        "best_epoch_exact": best_epoch,
        "best_step": best_step,
        "val_balanced_accuracy": val_bacc,
        "test_balanced_accuracy": test_bacc,
        "val_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "val_npz_path": val_npz_path,
        "test_npz_path": test_npz_path,
        "saved_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "hyperparams": hyperparams,
        "train_time_sec": train_time_sec,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "gpu_name": gpu_name,
    }

    json_path = os.path.join(save_dir, f"{task}_{model_name}.json")
    if os.path.exists(json_path):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = f"{json_path}.bak-{ts}"
        os.rename(json_path, backup_path)
        print(f"[warn] existing sleep sidecar json found, backed up to {backup_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if not os.path.exists(json_path):
        raise RuntimeError(f"save_sleep_epoch_results: failed to write {json_path}")
    with open(json_path) as f:
        json.load(f)  # 回读校验 JSON 没写坏

    print(f"Saved sleep val/test predictions to {val_npz_path} / {test_npz_path}")
    print(f"Saved sleep metadata to {json_path}")
    print(f"  best_epoch={meta['best_epoch']} val_bacc={val_bacc:.5f} test_bacc={test_bacc:.5f}")


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

        # [LOSO] 目前 motor6class（20折）和 stress（17折）支持 subject-independent 划分；
        # 不传 --split-mode（默认 random_epoch）时，其余分支行为完全不变。
        split_mode = config.get("split_mode", "random_epoch")
        if split_mode == "subject_independent" and dataset_name not in ("motor6class", "stress"):
            raise ValueError(
                f"--split-mode subject_independent is only supported for "
                f"--dataset-name motor6class/stress, got {dataset_name!r}"
            )

        if dataset_name == "stress" and split_mode == "subject_independent":
            # [LOSO] 17折 subject-independent 划分，从 downstream_path 下的
            # train/val/test 子目录里按受试者（chunk 文件名里的 SubNN）重新分组，
            # 不使用原来固定的 random_epoch train/val/test 划分。
            if not config.get("test_subject") or not config.get("val_subject"):
                raise ValueError(
                    "--split-mode subject_independent requires --test-subject and --val-subject"
                )
            matrix_p_path = config.get("matrix_p_path") or "tMatrix_22x30_stress.npy"
            if not os.path.isabs(matrix_p_path) and not os.path.exists(matrix_p_path):
                script_dir = os.path.dirname(os.path.realpath(__file__))
                script_dir = os.path.dirname(script_dir)  # 回到 neurogpt_finetune 目录
                matrix_p_path = os.path.join(script_dir, matrix_p_path.lstrip('./'))
            train_dataset, validation_dataset, test_dataset = prepare_stress_dataset_subject_independent(
                downstream_path,
                config["test_subject"],
                config["val_subject"],
                matrix_p_path=matrix_p_path,
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                gpt_only=not config["use_encoder"],
            )

        elif dataset_name in ["KaggleERN", "stress"]:

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

            if split_mode == "subject_independent":
                # [LOSO] 20折 subject-independent 划分，从 downstream_path 下的
                # train/val/test 子目录里按受试者重新分组，不使用原来固定的
                # train/val/test 划分。
                if not config.get("test_subject") or not config.get("val_subject"):
                    raise ValueError(
                        "--split-mode subject_independent requires --test-subject and --val-subject"
                    )
                train_dataset, validation_dataset, test_dataset = prepare_motor6class_dataset_subject_independent(
                    downstream_path,
                    config["test_subject"],
                    config["val_subject"],
                    matrix_p_path=matrix_p_path,
                    chunk_len=config["chunk_len"],
                    num_chunks=config["num_chunks"],
                    ovlp=config["chunk_ovlp"],
                    gpt_only=not config["use_encoder"],
                    normalization=config["do_normalization"],
                )
            else:
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

                train_dataset = Motor6ClassDataset(
                    train_files,
                    sample_keys=["inputs", "attention_mask"],
                    chunk_len=config["chunk_len"],
                    num_chunks=config["num_chunks"],
                    ovlp=config["chunk_ovlp"],
                    root_path=train_path,
                    matrix_p_path=matrix_p_path,
                    gpt_only=not config["use_encoder"],
                    normalization=config["do_normalization"],
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
                    normalization=config["do_normalization"],
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
                    normalization=config["do_normalization"],
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

        elif dataset_name == "sleep5class":
            # Sleep 5分类数据集
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
            matrix_p_path = config.get("matrix_p_path", "tMatrix_22x6_seed.npy")
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
            
            train_dataset = Sleep5ClassDataset(
                train_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=train_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

            validation_dataset = Sleep5ClassDataset(
                val_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=val_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
            )

            test_dataset = Sleep5ClassDataset(
                test_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=test_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
                # [Sleep] test_dataset 只在训练结束后被 trainer.predict() 用一次
                # （不像 validation_dataset 会被周期性训练中评估复用），可以直接
                # 打开 return_sample_id，不影响正常训练/评估流程。
                return_sample_id=True,
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
        load_best_model_at_end=True,
        save_total_limit=1,  # 只保留最新/最佳 checkpoint，避免磁盘爆掉
        save_strategy="steps",
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=True,
    )

    loso_mode = config.get("split_mode", "random_epoch") == "subject_independent"
    if loso_mode and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start_time = time.time()

    if config["do_train"]:
        trainer.train(resume_from_checkpoint=config["resume_from"])
        trainer.save_model(os.path.join(config["log_dir"], "model_final"))

    train_time_sec = time.time() - train_start_time

    if test_dataset is not None:
        test_prediction = trainer.predict(test_dataset)

        if loso_mode:
            # [LOSO] 保存 {task}_{model}_fold{i:02d}.npz/.json，让所有下游指标
            # 事后都能从这两个文件重新算，不需要重跑训练。不改动上面已有的
            # test_metrics.csv/test_predictions.npy 等既有保存逻辑。
            save_loso_fold_results(config, trainer, test_dataset, test_prediction, train_time_sec)
        elif config["dataset_name"] == "sleep5class":
            # [Sleep] 除了 test，还要对 val 集也做一次干净推理（trainer 内部已经是
            # load_best_model_at_end=True 之后的最佳权重，不用重新 reload），存
            # sleep_{model}_val.npz/_test.npz/sleep_{model}.json，供事后算
            # per-stage recall/confusion matrix/macro-F1/kappa/CI。val_dataset 复用
            # 上面构造 Sleep5ClassDataset 时用过的 val_files/val_path/matrix_p_path，
            # 但要单独建一份 return_sample_id=True 的实例——不能直接改
            # validation_dataset 本身，那个还要被 HF Trainer 训练期间的周期性 eval
            # 复用，混进 sample_id 字段可能打乱那条已经在用的路径。
            val_dataset_for_npz = Sleep5ClassDataset(
                val_files,
                sample_keys=["inputs", "attention_mask"],
                chunk_len=config["chunk_len"],
                num_chunks=config["num_chunks"],
                ovlp=config["chunk_ovlp"],
                root_path=val_path,
                matrix_p_path=matrix_p_path,
                gpt_only=not config["use_encoder"],
                return_sample_id=True,
            )
            val_prediction = trainer.predict(val_dataset_for_npz)
            save_sleep_epoch_results(
                config, trainer, val_dataset_for_npz, test_dataset,
                val_prediction, test_prediction, train_time_sec,
            )
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
        
        # 为所有数据集计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 获取预测和真实标签
        preds_logits = test_prediction.predictions
        labels = test_prediction.label_ids
        
        # 处理 chunk 级别预测的情况（与 make_decoding_accuracy_metrics 中的逻辑一致）
        # preds_logits 可能是 (batch_size * num_chunks, num_classes)
        # labels 是 (batch_size,)
        if len(preds_logits) != len(labels):
            logits_batch_size = len(preds_logits)
            labels_batch_size = len(labels)
            
            # 检查是否是 chunk 级别的情况
            if logits_batch_size % labels_batch_size == 0:
                num_chunks = logits_batch_size // labels_batch_size
                # Reshape preds_logits to (batch_size, num_chunks, num_classes)
                if preds_logits.ndim > 1:
                    preds_logits = preds_logits.reshape(labels_batch_size, num_chunks, -1)
                    # Average over chunks dimension to get (batch_size, num_classes)
                    preds_logits = preds_logits.mean(axis=1)
                else:
                    # 如果已经是类别索引，也需要 reshape 和平均
                    preds_logits = preds_logits.reshape(labels_batch_size, num_chunks)
                    # 对每个 sample 的 chunks 做多数投票
                    from collections import Counter
                    preds_aggregated = []
                    for i in range(labels_batch_size):
                        chunk_preds = preds_logits[i]
                        vote_counts = Counter(chunk_preds)
                        preds_aggregated.append(vote_counts.most_common(1)[0][0])
                    preds_logits = np.array(preds_aggregated)
            else:
                # 如果无法整除，尝试其他方法或截断
                import math
                gcd = math.gcd(logits_batch_size, labels_batch_size)
                if gcd > 1:
                    base_batch_size = gcd
                    logits_num_chunks = logits_batch_size // base_batch_size
                    if logits_num_chunks > 1 and preds_logits.ndim > 1:
                        preds_logits = preds_logits.reshape(base_batch_size, logits_num_chunks, -1)
                        preds_logits = preds_logits.mean(axis=1)
                        # 如果 labels 也需要处理
                        if labels_batch_size != base_batch_size:
                            labels_num_chunks = labels_batch_size // base_batch_size
                            if labels_num_chunks > 1:
                                # 对 labels 也做平均（虽然不太合理，但至少能运行）
                                labels = labels.reshape(base_batch_size, labels_num_chunks)
                                labels = labels[:, 0]  # 取第一个，因为同一 sample 的 label 应该相同
                    else:
                        # 截断到最小长度
                        min_len = min(logits_batch_size, labels_batch_size)
                        preds_logits = preds_logits[:min_len]
                        labels = labels[:min_len]
                else:
                    # 最后手段：截断
                    min_len = min(logits_batch_size, labels_batch_size)
                    preds_logits = preds_logits[:min_len]
                    labels = labels[:min_len]
        
        # 转换为类别预测
        if preds_logits.ndim > 1:
            # 如果是 logits，取 argmax
            y_pred = preds_logits.argmax(axis=-1)
        else:
            # 如果已经是类别索引
            y_pred = preds_logits
        
        y_true = labels
        
        # 确定类别数量
        num_classes = config["num_decoding_classes"]
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        
        # 保存混淆矩阵为CSV
        cm_df = pd.DataFrame(
            cm, 
            index=[f'True {i}' for i in range(num_classes)],
            columns=[f'Pred {i}' for i in range(num_classes)]
        )
        cm_df.to_csv(
            os.path.join(config["log_dir"], "test_confusion_matrix.csv")
        )
        
        # 保存混淆矩阵可视化
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[f'Class {i}' for i in range(num_classes)],
            yticklabels=[f'Class {i}' for i in range(num_classes)]
        )
        plt.title(f'Confusion Matrix - {config["dataset_name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(
            os.path.join(config["log_dir"], "test_confusion_matrix.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        print(f"✓ 混淆矩阵已保存到: test_confusion_matrix.csv 和 test_confusion_matrix.png")
        
        # 投票评估（仅对emotion7class数据集）
        if config["dataset_name"] == "emotion7class" and hasattr(test_dataset, 'epoch_ids'):
            print("\n=== 进行投票评估 ===")
            
            # 从dataset中收集所有epoch_ids（predict的顺序应该和dataset的顺序一致）
            sample_epoch_ids = test_dataset.epoch_ids
            n_samples = len(sample_epoch_ids)
            n_predictions = len(test_prediction.predictions)
            
            # 检测实际的chunks数量
            if n_predictions % n_samples == 0:
                actual_chunks_per_sample = n_predictions // n_samples
                
                # 为每个chunk的预测分配epoch_id
                chunk_epoch_ids = []
                chunk_labels = []
                chunk_predictions = []
                
                for i, epoch_id in enumerate(sample_epoch_ids):
                    # 为这个样本的所有chunks分配相同的epoch_id
                    for chunk_idx in range(actual_chunks_per_sample):
                        pred_idx = i * actual_chunks_per_sample + chunk_idx
                        chunk_epoch_ids.append(epoch_id)
                        
                        # 获取这个chunk的预测
                        if test_prediction.predictions.ndim == 1:
                            chunk_pred = test_prediction.predictions[pred_idx]
                        else:
                            chunk_pred = test_prediction.predictions[pred_idx].argmax()
                        chunk_predictions.append(chunk_pred)
                        
                        # label也应该重复
                        if i < len(test_prediction.label_ids):
                            chunk_labels.append(test_prediction.label_ids[i])
                
                # 如果label_ids的数量已经等于predictions，说明已经正确展开
                if len(test_prediction.label_ids) == n_predictions:
                    chunk_labels = test_prediction.label_ids
                elif len(test_prediction.label_ids) == n_samples:
                    # label_ids是样本级别的，需要展开到chunk级别
                    chunk_labels = []
                    for i, label in enumerate(test_prediction.label_ids):
                        for _ in range(actual_chunks_per_sample):
                            chunk_labels.append(label)
                    chunk_labels = np.array(chunk_labels)
                else:
                    # 其他情况，使用已有的label_ids
                    chunk_labels = test_prediction.label_ids[:n_predictions]
                
                # 使用展开后的predictions（argmax后的类别）
                chunk_predictions_array = np.array(chunk_predictions)
                
                # 使用展开后的predictions（chunk级别的类别预测）
                voting_metrics = compute_voting_metrics(
                    chunk_predictions_array,  # 使用已经argmax后的类别
                    chunk_labels,
                    chunk_epoch_ids
                )
                
                # 计算视频级别的预测和标签用于混淆矩阵
                from collections import Counter
                video_predictions = []
                video_labels = []
                video_groups = {}
                
                for i, epoch_id in enumerate(chunk_epoch_ids):
                    video_index = extract_video_index(epoch_id)
                    subject_id = extract_subject_id(epoch_id)
                    
                    if video_index is not None:
                        video_key = (subject_id, video_index)
                        if video_key not in video_groups:
                            video_groups[video_key] = {'preds': [], 'label': None}
                        video_groups[video_key]['preds'].append(int(chunk_predictions_array[i]))
                        video_groups[video_key]['label'] = int(chunk_labels[i])
                
                # 进行投票得到视频级别的预测
                for video_key, video_data in video_groups.items():
                    preds = video_data['preds']
                    true_label = video_data['label']
                    
                    if len(preds) == 0:
                        continue
                    
                    # 多数投票
                    vote_counts = Counter(preds)
                    max_votes = max(vote_counts.values())
                    majority_classes = [cls for cls, count in vote_counts.items() if count == max_votes]
                    
                    if len(majority_classes) == 1:
                        video_pred = majority_classes[0]
                    else:
                        # 平票情况：如果真实标签在候选中，选择它；否则选择第一个
                        if true_label in majority_classes:
                            video_pred = true_label
                        else:
                            video_pred = majority_classes[0]
                    
                    video_predictions.append(video_pred)
                    video_labels.append(true_label)
                
                video_predictions = np.array(video_predictions)
                video_labels = np.array(video_labels)
                
                # 计算混淆矩阵
                from sklearn.metrics import confusion_matrix
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                num_classes = 6  # emotion 6分类（移除neutral后）
                cm = confusion_matrix(video_labels, video_predictions, labels=list(range(num_classes)))
                
                # 保存混淆矩阵为CSV
                cm_df = pd.DataFrame(
                    cm, 
                    index=[f'True {i}' for i in range(num_classes)],
                    columns=[f'Pred {i}' for i in range(num_classes)]
                )
                cm_df.to_csv(
                    os.path.join(config["log_dir"], "test_voting_confusion_matrix.csv")
                )
                
                # 保存混淆矩阵可视化
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=[f'Class {i}' for i in range(num_classes)],
                    yticklabels=[f'Class {i}' for i in range(num_classes)]
                )
                plt.title('Confusion Matrix - Video Level Voting')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(config["log_dir"], "test_voting_confusion_matrix.png"),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
            else:
                # 如果无法整除，说明有其他问题
                print(f"  ⚠️  警告: predictions数量 ({n_predictions}) 不是样本数量 ({n_samples}) 的整数倍")
                print(f"  尝试使用前 {n_samples} 个predictions进行投票评估")
                
                # 只使用前n_samples个predictions，对应的epoch_ids和labels
                voting_metrics = compute_voting_metrics(
                    test_prediction.predictions[:n_samples],
                    test_prediction.label_ids[:n_samples],
                    sample_epoch_ids[:n_samples]
                )
            
            # 保存投票评估结果
            voting_metrics_df = pd.DataFrame([voting_metrics], index=[0])
            voting_metrics_df.to_csv(
                os.path.join(config["log_dir"], "test_voting_metrics.csv"), 
                index=False
            )
            
            # 保存epoch_ids供后续分析（保存chunk级别的epoch_ids）
            if 'chunk_epoch_ids' in locals():
                np.save(
                    os.path.join(config["log_dir"], "test_epoch_ids.npy"),
                    np.array(chunk_epoch_ids, dtype=object),
                )
            else:
                np.save(
                    os.path.join(config["log_dir"], "test_epoch_ids.npy"),
                    np.array(sample_epoch_ids, dtype=object),
                )
            
            print(f"✓ 投票评估结果已保存到: test_voting_metrics.csv")
            if 'chunk_epoch_ids' in locals():
                print(f"✓ 混淆矩阵已保存到: test_voting_confusion_matrix.csv 和 test_voting_confusion_matrix.png")
            print(f"  视频级别准确率: {voting_metrics['video_accuracy']:.4f}")
            print(f"  视频级别BACC: {voting_metrics['video_bacc']:.4f}")
            print(f"  评估的视频数量: {voting_metrics['num_videos']}")
            if 'avg_subject_accuracy' in voting_metrics:
                print(f"  平均Subject准确率: {voting_metrics['avg_subject_accuracy']:.4f}")

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
        choices=("KaggleERN", "stress", "motor6class", "emotion7class", "sleep5class"),
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

    # ===== [LOSO] subject-independent LOSO（--dataset-name motor6class 支持20折，stress 支持17折）=====
    # 不传 --split-mode（默认 random_epoch）时，行为与之前完全一致。
    parser.add_argument(
        "--split-mode",
        metavar="STR",
        default="random_epoch",
        choices=("random_epoch", "subject_independent"),
        type=str,
        help="random_epoch(默认)=旧的固定 train/val/test 目录划分; "
        "subject_independent=LOSO 单折训练，需要 --test-subject/--val-subject "
        "（--dataset-name motor6class/stress 支持）。",
    )
    parser.add_argument(
        "--test-subject",
        metavar="STR",
        default=None,
        type=str,
        help="e.g. Sub04；--split-mode subject_independent 时必填",
    )
    parser.add_argument(
        "--val-subject",
        metavar="STR",
        default=None,
        type=str,
        help="e.g. Sub05；--split-mode subject_independent 时必填",
    )
    parser.add_argument(
        "--fold-idx",
        metavar="INT",
        default=0,
        type=int,
        help="LOSO fold 序号（0-based），用于保存文件名 {task}_{model}_fold{i:02d}",
    )
    parser.add_argument(
        "--model-name",
        metavar="STR",
        default="neurogpt",
        type=str,
        help="保存 {task}_{model}_fold{i}.npz/json 时使用的模型名",
    )
    parser.add_argument(
        "--task-name",
        metavar="STR",
        default=None,
        type=str,
        help="保存 {task}_{model}_fold{i}.npz/json 时使用的任务名；默认取 --dataset-name",
    )
    parser.add_argument(
        "--fold-results-dir",
        metavar="DIR",
        default=None,
        type=str,
        help="npz/json 保存目录；默认使用 --log-dir",
    )

    return parser


if __name__ == "__main__":

    trainer = train()
