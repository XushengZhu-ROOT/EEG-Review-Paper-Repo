# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
from pyexpat import model
import numpy as np
import re
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys
import yaml

from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from scipy import interpolate
import modeling_finetune

import psutil
import GPUtil

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    parser.add_argument('--classifier_window_size', default=1, type=int, # motor 
                        help='time window size for classifier [labram_base_patch200_200_cbramod3lyclassifier]')
    parser.add_argument('--classifier_dropout', default=0.1, type=float,
                        help='dropout for layers of classifier [labram_base_patch200_200_cbramod3lyclassifier]')


    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze all weights except the classification head (model.head)')
    parser.add_argument('--dataset_path', default='',
                        help='path to dataset')                        
    parser.add_argument('--channel_size', default=20, type=int,
                        help='number of the classification types')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')
    parser.add_argument('--pos_weight', default=None, type=float)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV')

    # ===== [LOSO] 20-fold subject-independent LOSO（目前仅 --dataset Motor 支持）=====
    # 不传 --split_mode（默认 random_epoch）时，行为与之前完全一致。
    parser.add_argument('--split_mode', type=str, default='random_epoch',
                        choices=['random_epoch', 'subject_independent'],
                        help='random_epoch(默认)=旧的固定 train/val/test 目录划分；'
                             'subject_independent=20折 LOSO 单折训练，需要 --test_subject/--val_subject'
                             '（目前仅 --dataset Motor 支持）。')
    parser.add_argument('--test_subject', type=str, default=None,
                        help='e.g. Sub04；split_mode=subject_independent 时必填')
    parser.add_argument('--val_subject', type=str, default=None,
                        help='e.g. Sub05；split_mode=subject_independent 时必填')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='LOSO fold 序号（0-based），用于保存文件名 {task}_{model}_fold{i:02d}')
    parser.add_argument('--model_name', type=str, default='labram',
                        help='保存 {task}_{model}_fold{i}.npz/json 时使用的模型名')
    parser.add_argument('--task_name', type=str, default=None,
                        help='保存 {task}_{model}_fold{i}.npz/json 时使用的任务名；默认取 --dataset 的小写')
    parser.add_argument('--fold_results_dir', type=str, default=None,
                        help='npz/json 保存目录；默认使用 --output_dir')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def get_models(args):
    if 'cbramod3lyclassifier' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            channel_size=args.channel_size,
            window_size=args.classifier_window_size,
            classifier_dropout=args.classifier_dropout,
        )
    elif 'returnpatchtoken' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            channel_size=args.channel_size,
            window_size=args.classifier_window_size,
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
        )

    return model


def get_dataset(args):
    if getattr(args, 'split_mode', 'random_epoch') == 'subject_independent' and args.dataset not in ('Motor', 'Stress'):
        raise ValueError(
            f"--split_mode subject_independent is only supported for --dataset Motor/Stress, got {args.dataset!r}"
        )
    if args.dataset == 'KaggleERN':
        train_dataset, test_dataset, val_dataset = utils.prepare_KaggleERN_dataset(args.dataset_path)
        ch_names = ['FP1', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', \
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', \
            'PO7', 'POZ', 'O1', 'O2']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]


    elif args.dataset == 'Motor':
        split_mode = getattr(args, 'split_mode', 'random_epoch')
        if split_mode == 'subject_independent':
            if not args.test_subject or not args.val_subject:
                raise ValueError("--split_mode subject_independent requires --test_subject and --val_subject")
            train_dataset, test_dataset, val_dataset = utils.prepare_Motor_dataset_subject_independent(
                args.dataset_path, args.test_subject, args.val_subject)
        else:
            train_dataset, test_dataset, val_dataset = utils.prepare_Motor_dataset(args.dataset_path)
        # ch_names = ['F7', 'Fp1', 'Fp2', 'F8', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'P8', 'P7', 'Pz', 'P4', 'T3', 'P3', 'O1', 'O2', 'C4', 'T4', 'A2']
        ch_names = ['F7','FP1','FP2','F8','F3','FZ','F4','C3','CZ','P8','P7','PZ','P4','T3','P3','O1','O2','C4','T4','A2']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        # metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        # Motor 是多分类任务，使用多分类指标（pr_auc 和 roc_auc 在多分类中不支持）
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == 'Seed':
        train_dataset, test_dataset, val_dataset = utils.prepare_Seed_dataset(args.dataset_path)
        # 62通道名列表（从emotion实验日志.txt获取）
        ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6  # 6分类任务（排除neutral类别）
        # Seed是6情绪分类任务（happy, sad, disgust, fear, surprise, anger），使用多分类指标
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == 'Sleep':
        train_dataset, test_dataset, val_dataset = utils.prepare_Sleep_dataset(args.dataset_path)
        # 6通道：C3, C4, F3, F4, O1, O2
        ch_names = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 5  # 5分类任务（0,1,2,3,4）
        # Sleep是5分类任务，使用多分类指标
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    
    elif args.dataset == 'Stress':
        split_mode = getattr(args, 'split_mode', 'random_epoch')
        if split_mode == 'subject_independent':
            if not args.test_subject or not args.val_subject:
                raise ValueError("--split_mode subject_independent requires --test_subject and --val_subject")
            train_dataset, test_dataset, val_dataset = utils.prepare_Stress_dataset_subject_independent(
                args.dataset_path, args.test_subject, args.val_subject)
        else:
            train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset(args.dataset_path)
        ch_names = []
        if args.channel_size == 30:
            ch_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']
        elif args.channel_size == 20:
            ch_names= ['FP1','F7','F3','F8','FZ','FC4', 'FT8', 'T3', 'C3', 'CZ', 'T4', 'TP7', 'CP3', 'CPZ', 'CP4','T5', 'P3', 'PZ', 'P4', 'T6']
        elif args.channel_size == 11:
            ch_names = ['FP1','FP2','F3','FZ','F4','FC3','FCZ','FC4','C3','CZ','C4']
        elif args.channel_size == 16:
            ch_names = ['C4', 'O2', 'P3', 'FP1', 'F7', 'FP2', 'P4', 'O1', 'F8', 'F3', 'C3', 'F4', 'FT8', 'FT7', 'TP7', 'TP8']
        else:
            raise ValueError(f"Undefined channel size: {args.channel_size}")
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def save_loso_fold_results(args, model_without_ddp, device, dataset_test, ch_names,
                            best_epoch, train_time_sec, n_parameters):
    """[LOSO] 用调用方已经载入 model_without_ddp 的最佳验证 epoch 权重，对 test 集重新做一次
    干净的推理（带 sample_id），保存：
      {task}_{model}_fold{i:02d}.npz  -- sample_id / y_true / y_pred / y_prob / subject_id
                                          （y_prob 恒为 (N,2)：Motor 是 softmax 输出，
                                          Stress 是 [1-sigmoid, sigmoid]，与 cbramod/neurolm
                                          Stress LOSO 的 npz schema 一致，供 aggregate 脚本用
                                          y_prob[:,1] 算 roc_auc/pr_auc）
      {task}_{model}_fold{i:02d}.json -- fold / test_subject / val_subject / balanced_accuracy /
                                          best_epoch / hyperparams / train_time_sec /
                                          peak_gpu_mem_mb / gpu_name
    不改动训练逻辑/模型定义，只是复用已经训练好的模型做一次推理并落盘。
    任何失败（没有样本、写文件失败、写完读不回来）都直接抛异常，不静默跳过。
    """
    model_without_ddp.eval()
    input_chans = utils.get_input_chans(ch_names) if ch_names is not None else None

    # 复用当前 fold 的 test 文件列表（dataset_test.root / .files），重新构造一个
    # return_sample_id=True 的 Loader，与训练/评估时用的是同一批底层 pickle 文件。
    # Motor 是 6 分类（MotorLoader，softmax+argmax），Stress 是二分类（StressLoader，
    # 模型输出单个 logit，sigmoid+0.5 阈值），两者的推理分支不同，其余保存/校验逻辑共用。
    is_binary = args.dataset == 'Stress'
    loader_cls = utils.StressLoader if is_binary else utils.MotorLoader
    sid_dataset = loader_cls(
        dataset_test.root, dataset_test.files,
        sampling_rate=dataset_test.sampling_rate,
        data_key=dataset_test.data_key, label_key=dataset_test.label_key,
        expected_channels=dataset_test.expected_channels,
        return_sample_id=True,
    )
    sid_loader = torch.utils.data.DataLoader(
        sid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=utils.skip_failed_collate,
    )

    # T 按已知任务时长动态算出 patch 数 A = T // 200（与 engine_for_finetuning.py 的
    # evaluate() 用同一份白名单，覆盖 Motor T=200 / Stress T=1000 等）。
    _VALID_T = {200, 600, 800, 1000, 6000}
    subj_re = re.compile(r'^S(\d+)_')
    sample_ids, y_true, y_pred, y_prob, subject_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in sid_loader:
            if batch is None:
                continue
            X, Y, batch_sample_ids = batch
            X = X.float().to(device, non_blocking=True) / 100
            if len(X.shape) != 3:
                raise ValueError(f"save_loso_fold_results: unexpected input shape {X.shape}")
            B, N, T = X.shape
            if T not in _VALID_T:
                raise ValueError(f"save_loso_fold_results: unexpected time dimension T={T}")
            X = X.view(B, N, T // 200, 200)

            output = model_without_ddp(X, input_chans=input_chans)
            if is_binary:
                pos_prob = torch.sigmoid(output).cpu().view(-1)
                probs = torch.stack([1.0 - pos_prob, pos_prob], dim=1)
                preds = (pos_prob > 0.5).long()
            else:
                probs = torch.softmax(output, dim=-1).cpu()
                preds = torch.argmax(output, dim=-1).cpu()
            for i, sid in enumerate(batch_sample_ids):
                m = subj_re.match(sid)
                if not m:
                    raise ValueError(f"Cannot parse subject_id from sample_id: {sid!r}")
                sample_ids.append(sid)
                y_true.append(int(Y[i]))
                y_pred.append(int(preds[i].item()))
                y_prob.append(probs[i].numpy())
                subject_ids.append(int(m.group(1)))

    if len(sample_ids) == 0:
        raise RuntimeError("save_loso_fold_results: no samples collected, refusing to save an empty file.")

    sample_ids_arr = np.array(sample_ids)
    order = np.argsort(sample_ids_arr)
    sample_ids_arr = sample_ids_arr[order]
    y_true_arr = np.array(y_true, dtype=np.int64)[order]
    y_pred_arr = np.array(y_pred, dtype=np.int64)[order]
    y_prob_arr = np.array(y_prob, dtype=np.float32)[order]
    subject_id_arr = np.array(subject_ids, dtype=np.int64)[order]

    task = args.task_name or args.dataset.lower()
    model_name = args.model_name or "labram"
    fold_idx = args.fold_idx
    save_dir = args.fold_results_dir or args.output_dir
    if not save_dir:
        raise ValueError("save_loso_fold_results requires --fold_results_dir or --output_dir to be set")
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(save_dir, f"{task}_{model_name}_fold{fold_idx:02d}.npz")
    json_path = os.path.join(save_dir, f"{task}_{model_name}_fold{fold_idx:02d}.json")

    # 已有旧结果先改名备份，绝不静默覆盖
    for path in (npz_path, json_path):
        if os.path.exists(path):
            ts = time.strftime("%Y%m%d-%H%M%S")
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
        if len(_reload[key]) != len(sample_ids_arr):
            raise RuntimeError(f"save_loso_fold_results: {npz_path} key {key!r} length mismatch after write")

    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    hyperparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.opt,
        "seed": args.seed,
        "layer_decay": args.layer_decay,
        "warmup_epochs": args.warmup_epochs,
        "drop_path": args.drop_path,
        "smoothing": args.smoothing,
        "freeze_backbone": bool(args.freeze_backbone),
        "channel_size": args.channel_size,
        "model": args.model,
        "n_parameters": n_parameters,
    }

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated(0) / (1024 ** 2)) if torch.cuda.is_available() else None

    meta = {
        "fold": fold_idx,
        "test_subject": args.test_subject,
        "val_subject": args.val_subject,
        "balanced_accuracy": balanced_accuracy,
        "best_epoch": best_epoch,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
    print(f"  fold={fold_idx} test={args.test_subject} val={args.val_subject} "
          f"best_epoch={best_epoch} balanced_accuracy={balanced_accuracy:.5f}")


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    if args.output_dir and utils.is_main_process():
        args_dict = vars(args)
        args_path = os.path.join(args.output_dir, "run_config.yaml")
        try:
            with open(args_path, 'w', encoding='utf-8') as f:
                # 將 args (Namespace) 轉換為 dict 再儲存
                yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)
            print(f"Arguments configuration saved to {args_path}")
        except Exception as e:
            print(f"Warning: Error saving arguments to YAML: {e}")

    # 自动检测设备：优先 CUDA，然后 MPS（Mac GPU），最后 CPU
    if args.device == 'cuda':
        if torch.cuda.is_available():
            args.device = 'cuda'
            print("Using CUDA device")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
            print("CUDA not available, using MPS (Mac GPU) instead")
        else:
            args.device = 'cpu'
            print("CUDA and MPS not available, using CPU instead")
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        args.device = 'mps'
        print("Using MPS (Mac GPU)")
    device = torch.device(args.device)

    
    # 初始化資源監控
    training_start_time = time.time()
    gpu_logs = []
    
    def log_gpu_usage(epoch, step=None):
        """記錄GPU使用情況"""
        if torch.cuda.is_available():
            gpu_stats = {
                'epoch': epoch,
                'step': step,
                'timestamp': time.time() - training_start_time,
            }
            
            # 使用 torch 獲取GPU信息
            for i in range(torch.cuda.device_count()):
                gpu_stats[f'gpu_{i}_memory_allocated_GB'] = torch.cuda.memory_allocated(i) / 1024**3
                gpu_stats[f'gpu_{i}_memory_reserved_GB'] = torch.cuda.memory_reserved(i) / 1024**3
                gpu_stats[f'gpu_{i}_max_memory_allocated_GB'] = torch.cuda.max_memory_allocated(i) / 1024**3
            
            # 使用 GPUtil 獲取更詳細信息（如果可用）
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_stats[f'gpu_{i}_utilization_%'] = gpu.load * 100
                    gpu_stats[f'gpu_{i}_temperature_C'] = gpu.temperature
                    gpu_stats[f'gpu_{i}_memory_used_GB'] = gpu.memoryUsed / 1024
                    gpu_stats[f'gpu_{i}_memory_total_GB'] = gpu.memoryTotal / 1024
            except:
                pass
            
            # CPU和系統資源
            gpu_stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            gpu_stats['ram_used_GB'] = psutil.virtual_memory().used / 1024**3
            gpu_stats['ram_percent'] = psutil.virtual_memory().percent
            
            gpu_logs.append(gpu_stats)
            return gpu_stats
        return None
    
    # 記錄訓練開始時的GPU狀態
    initial_gpu_stats = log_gpu_usage(epoch=-1, step=0)
    if initial_gpu_stats and utils.is_main_process():
        print("\n" + "="*50)
        print("Initial GPU Status:")
        for key, value in initial_gpu_stats.items():
            if key not in ['epoch', 'step', 'timestamp']:
                print(f"  {key}: {value:.2f}")
        print("="*50 + "\n")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
    collate_fn=utils.skip_failed_collate,)


    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=utils.skip_failed_collate,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                collate_fn=utils.skip_failed_collate,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                collate_fn=utils.skip_failed_collate,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Handle time_embed extension: if pretrained has 16 time windows and model has 32, 
        # load first 16 and initialize the rest (copy from last time window or zero init)
        if 'time_embed' in checkpoint_model and 'time_embed' in state_dict:
            pretrained_time_embed = checkpoint_model['time_embed']
            model_time_embed = state_dict['time_embed']
            if pretrained_time_embed.shape[1] == 16 and model_time_embed.shape[1] == 32:
                print(f"Extending time_embed from 16 to 32 time windows")
                # Create extended time_embed: first 16 from pretrained, rest initialized
                extended_time_embed = torch.zeros_like(model_time_embed)
                extended_time_embed[:, :16, :] = pretrained_time_embed
                # Initialize remaining time windows by copying the last pretrained time window
                # This provides a reasonable starting point for the new time windows
                extended_time_embed[:, 16:, :] = pretrained_time_embed[:, -1:, :].expand(-1, 16, -1)
                checkpoint_model['time_embed'] = extended_time_embed
                print(f"Extended time_embed: loaded first 16 from pretrained, initialized rest from last time window")

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    # freeze backbone
    if args.freeze_backbone:
        if utils.is_main_process():
            print("\n" + "="*25 + " FREEZING BACKBONE " + "="*25)
            print("All model parameters are frozen EXCEPT for the classifier head.")
	
        for param in model.parameters():
            param.requires_grad = False
	
        unfrozen_parts = []
	
        # unfreeze fc_norm
        if hasattr(model, 'fc_norm'):
            for param in model.fc_norm.parameters():
                param.requires_grad = True
            unfrozen_parts.append("'model.fc_norm'")

        # unfreeze head
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
            unfrozen_parts.append("'model.head'")

        if utils.is_main_process():
            if not unfrozen_parts:
                print("WARNING: Model does not have 'head' or 'fc_norm' attributes. No parameters were unfrozen.")
            else:
                parts_str = " and ".join(unfrozen_parts)
                print(f"Successfully unfroze {parts_str} parameters for training.")
            print("="*70 + "\n")


    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    if args.output_dir and utils.is_main_process():
        model_structure_path = os.path.join(args.output_dir, "model_structure.txt")
        try:
            with open(model_structure_path, 'w', encoding='utf-8') as f:
                # model_without_ddp 尚未被 DDP or DeepSpeed 封裝
                f.write(str(model_without_ddp))
                f.write("\n\n")
                f.write(f"Total Trainable Parameters: {n_parameters}\n")
            print(f"Model structure saved to {model_structure_path}")
        except Exception as e:
            print(f"Warning: Error saving model structure: {e}")

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Physical batch size per GPU = %d" % args.batch_size)
    print("Update frequency (gradient accumulation steps) = %d" % args.update_freq)
    print("Effective batch size = %d (physical_batch_size × update_freq × num_gpus)" % total_batch_size)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)
    print("Note: Gradients are accumulated over %d batches before each optimizer update" % args.update_freq)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    # Debug: 打印调度器参数
    print(f"[DEBUG] Cosine scheduler parameters:")
    print(f"  - epochs: {args.epochs}")
    print(f"  - num_training_steps_per_epoch: {num_training_steps_per_epoch}")
    print(f"  - warmup_epochs: {args.warmup_epochs}")
    print(f"  - warmup_steps: {args.warmup_steps}")
    print(f"  - total_steps = epochs * niter_per_ep = {args.epochs} * {num_training_steps_per_epoch} = {args.epochs * num_training_steps_per_epoch}")
    
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    print(f"[DEBUG] LR schedule values length: {len(lr_schedule_values)}")
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    print(f"[DEBUG] Weight decay scheduler parameters:")
    print(f"  - weight_decay: {args.weight_decay}")
    print(f"  - weight_decay_end: {args.weight_decay_end}")
    print(f"  - epochs: {args.epochs}")
    print(f"  - num_training_steps_per_epoch: {num_training_steps_per_epoch}")
    
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print(f"[DEBUG] WD schedule values length: {len(wd_schedule_values)}")
    print(f"[DEBUG] WD schedule values (first 5): {wd_schedule_values[:5] if len(wd_schedule_values) > 0 else 'EMPTY'}")
    print(f"[DEBUG] WD schedule values (last 5): {wd_schedule_values[-5:] if len(wd_schedule_values) > 0 else 'EMPTY'}")
    
    if len(wd_schedule_values) > 0:
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    else:
        print("[ERROR] wd_schedule_values is empty! This usually means num_training_steps_per_epoch is 0 or too small.")
        print("[ERROR] Please check:")
        print(f"  - Dataset size: {len(dataset_train)}")
        print(f"  - Total batch size: {total_batch_size}")
        print(f"  - num_training_steps_per_epoch = {len(dataset_train)} // {total_batch_size} = {num_training_steps_per_epoch}")

    if args.nb_classes == 1:
        if args.pos_weight:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
            
    if args.eval:
        balanced_accuracy = []
        accuracy = []
        for data_loader in data_loader_test:
            test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1))
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_val_bacc = 0.0
    max_test_bacc = 0.0

    # ===== [LOSO] 按验证集 balanced_accuracy 选出的最佳 epoch（20折 subject_independent
    # 用）。上面 checkpoint-best.pth 的选择标准现在也已统一改为 val_bacc，
    # 这里的记账逻辑因此变为冗余但仍然正确（两者选出的应是同一个 epoch），保留不动。=====
    loso_mode = getattr(args, 'split_mode', 'random_epoch') == 'subject_independent'
    best_val_bacc = -1.0
    best_epoch_loso = 0
    best_model_state_loso = None
    if loso_mode and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        
        epoch_start_time = time.time()
        # 記錄epoch開始時的GPU狀態
        log_gpu_usage(epoch=epoch, step=0)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
            ch_names=ch_names, is_binary=args.nb_classes == 1
        )
        
        
        epoch_time = time.time() - epoch_start_time
        # 記錄epoch結束時的GPU狀態
        gpu_stats = log_gpu_usage(epoch=epoch, step='end')
        
        # 打印epoch時間和GPU使用情況
        if utils.is_main_process():
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} completed in {epoch_time/60:.2f} minutes ({epoch_time:.2f} seconds)")
            if gpu_stats:
                print(f"GPU Memory Usage:")
                for i in range(torch.cuda.device_count()):
                    if f'gpu_{i}_memory_allocated_GB' in gpu_stats:
                        print(f"  GPU {i}: {gpu_stats[f'gpu_{i}_memory_allocated_GB']:.2f} GB allocated, "
                              f"{gpu_stats[f'gpu_{i}_max_memory_allocated_GB']:.2f} GB max")
                    if f'gpu_{i}_utilization_%' in gpu_stats:
                        print(f"  GPU {i} Utilization: {gpu_stats[f'gpu_{i}_utilization_%']:.1f}%")
            print(f"{'='*60}\n")

        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")
            
            if max_val_bacc < val_stats["balanced_accuracy"]:
                max_val_bacc = val_stats["balanced_accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                max_test_bacc = test_stats["balanced_accuracy"]

            # ===== [LOSO] 按验证集 balanced_accuracy 记录最佳 epoch 的模型权重（内存中），
            # 用于训练结束后重新对 test 集做一次干净的推理，事后保存 npz/json。=====
            if loso_mode:
                current_val_bacc = val_stats.get('balanced_accuracy')
                if current_val_bacc is not None and current_val_bacc > best_val_bacc:
                    best_val_bacc = current_val_bacc
                    best_epoch_loso = epoch + 1
                    best_model_state_loso = {k: v.clone().cpu() for k, v in model_without_ddp.state_dict().items()}

            print(f'Max val bacc: {max_val_bacc:.2f}%, max test bacc: {max_test_bacc:.2f}%')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
            # 保存混淆矩阵到log（仅对多分类任务）
            if args.nb_classes > 1 and 'confusion_matrix' in test_stats:
                log_stats['test_confusion_matrix'] = test_stats['confusion_matrix']
                # 打印混淆矩阵（用于快速检查）
                if utils.is_main_process():
                    print(f"\n测试集混淆矩阵 (Epoch {epoch}):")
                    cm = np.array(test_stats['confusion_matrix'])
                    print(cm)
                    # 计算每个类别的准确率
                    print("\n各类别准确率:")
                    for i in range(len(cm)):
                        if cm[i].sum() > 0:
                            class_acc = cm[i, i] / cm[i].sum() * 100
                            print(f"  Class {i}: {class_acc:.2f}% ({cm[i, i]}/{cm[i].sum()})")
                    print()
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # ===== [LOSO] 用被选中的最佳 epoch 权重（不是训练循环结束时留在 model 里的最后一轮权重）
    # 对 test 集重新推理一次，保存 {task}_{model}_fold{i:02d}.npz/.json，
    # 让所有下游指标事后都能从这两个文件重新算，不需要重跑训练。
    # 保存失败直接抛异常退出，不静默跳过。=====
    if loso_mode and utils.is_main_process():
        if best_model_state_loso is None:
            raise RuntimeError(
                "LOSO fold training finished but best_model_state_loso is None "
                "(val balanced_accuracy was never recorded) -- check --epochs / dataset."
            )
        model_without_ddp.load_state_dict(best_model_state_loso)
        model_without_ddp.to(device)
        save_loso_fold_results(
            args, model_without_ddp, device, dataset_test, ch_names,
            best_epoch_loso, total_time, n_parameters,
        )

    # 保存詳細的資源使用報告
    if args.output_dir and utils.is_main_process():
        # 計算平均GPU使用率
        if gpu_logs:
            import pandas as pd
            df = pd.DataFrame(gpu_logs)
            
            # 生成摘要統計
            summary = {
                'total_training_time_seconds': time.time() - training_start_time,
                'total_training_time_readable': str(datetime.timedelta(seconds=int(time.time() - training_start_time))),
                'total_epochs': args.epochs,
                'average_time_per_epoch_seconds': (time.time() - training_start_time) / args.epochs,
            }
            
            # GPU統計
            for i in range(torch.cuda.device_count()):
                if f'gpu_{i}_memory_allocated_GB' in df.columns:
                    summary[f'gpu_{i}_avg_memory_allocated_GB'] = df[f'gpu_{i}_memory_allocated_GB'].mean()
                    summary[f'gpu_{i}_max_memory_allocated_GB'] = df[f'gpu_{i}_max_memory_allocated_GB'].max()
                if f'gpu_{i}_utilization_%' in df.columns:
                    summary[f'gpu_{i}_avg_utilization_%'] = df[f'gpu_{i}_utilization_%'].mean()
                    summary[f'gpu_{i}_max_utilization_%'] = df[f'gpu_{i}_utilization_%'].max()
            
            # CPU和RAM統計
            if 'cpu_percent' in df.columns:
                summary['avg_cpu_percent'] = df['cpu_percent'].mean()
                summary['max_cpu_percent'] = df['cpu_percent'].max()
            if 'ram_used_GB' in df.columns:
                summary['avg_ram_used_GB'] = df['ram_used_GB'].mean()
                summary['max_ram_used_GB'] = df['ram_used_GB'].max()
            
            # 保存摘要
            with open(os.path.join(args.output_dir, "resource_summary.json"), "w") as f:
                json.dump(summary, f, indent=4)
            
            # 保存詳細日誌
            df.to_csv(os.path.join(args.output_dir, "resource_logs.csv"), index=False)
            
            # 打印摘要
            print("\n" + "="*60)
            print("Training Resource Summary:")
            print("="*60)
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
            print("="*60 + "\n")

if __name__ == '__main__':
    opts, ds_init = get_args()
    
    # print("\n===== opts 内容 =====")
    # print(opts)
    # print("\n===== ds_init 内容 =====")
    # print(ds_init)

    # 保存当前的 opts 配置到本目录
    # import yaml
    # with open("opts_saved.yaml", "w") as f:
    #     yaml.safe_dump(vars(opts), f, sort_keys=False)
    # print("opts 已保存到 opts_saved.yaml")


    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
