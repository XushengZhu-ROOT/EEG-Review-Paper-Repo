import os
import argparse
import pickle
import yaml
import json
import re
import time as _time

# [GPU 环境兼容] PyTorch Lightning 的 TensorBoardLogger 底层用
# torch.utils.tensorboard.SummaryWriter，它会优先尝试 `import tensorflow`
# （只是为了拿 tf.io.gfile，不是真的要用 TF 训练）。在有 cgroup CPU 配额限制的
# 容器/共享 GPU 环境里（比如 DSMLP），TF 初始化时读到的可调度 CPU 数经常远大于
# 实际配额，会尝试开一大堆线程，观察到的现象是进程卡住不动、CPU 占用不高不低
# （因为在线程调度上空转），Ctrl+C 也没反应，但 GPU 显存/利用率还是 0——发生在
# 任何训练开始之前，因为 TensorBoardLogger 在 Trainer 里几乎是第一步就会初始化。
# 必须在 tensorboard/pytorch_lightning 被 import 之前设置这个标记，强制
# tensorboard 用它自带的 numpy-based stub 而不是真的 import tensorflow。
import tensorboard.compat
tensorboard.compat.notf = True

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)


def binary_metrics_fn(y_true, y_pred_proba, metrics=None, threshold=0.5):
    """
    使用 sklearn 实现二分类指标计算
    替换 pyhealth.metrics.binary_metrics_fn
    
    Args:
        y_true: 真实标签 (1D array)
        y_pred_proba: 预测概率 (1D array)
        metrics: 要计算的指标列表
        threshold: 二分类阈值
    
    Returns:
        dict: 包含各项指标的字典
    """
    if metrics is None:
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    
    # 使用阈值转换为二分类预测
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    result = {}
    
    if "accuracy" in metrics:
        result["accuracy"] = accuracy_score(y_true, y_pred)
    
    if "balanced_accuracy" in metrics:
        result["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
    if "roc_auc" in metrics:
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            # [LOSO] Stress 某一折的 val/test 受试者可能只有单一类别（17 个受试者
            # 里有 11 个只做过 increase 或只做过 normal），roc_auc 在数学上未定义，
            # 返回 NaN 而不是 0.0——跟 cbramod_finetune/finetune_evaluator.py 的
            # get_metrics_for_binaryclass 是同一套处理方式，NaN 才能被下游
            # aggregate_loso_results_stress.py 正确识别为"这折算不出来"而不是
            # "算出来是 0"。
            result["roc_auc"] = float("nan")

    if "pr_auc" in metrics:
        try:
            result["pr_auc"] = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            result["pr_auc"] = float("nan")
    
    return result

from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
    Ada_BIOT,
    Labram_style_BIOTClassifier,
    Labram_style_Ada_BIOT,
    CBraMod_3lyStyle_LayerNorm_BIOT,
    CBraMod_3lyStyle_LayerNorm_Ada_BIOT
)
from utils import (
    KaggleERNLoader, TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE,
    StressLoader, collate_fn_stress_with_sample_id, list_stress_files_by_subject,
    collate_fn_kaggleern_with_sample_id,
)


def _binary_degenerate(gt):
    """gt 全为 0 或全为 1（LOSO 下某折的 val/test 受试者只做过 increase 或只做过
    normal 时会发生，17 个受试者里有 11 个是这种情况）。"""
    total_pos = sum(gt)
    return total_pos * (len(gt) - total_pos) == 0


def _compute_binary_metrics(gt, result, threshold):
    """[LOSO 修复] 统一走 binary_metrics_fn，不再对退化 batch（gt 全 0/全 1）
    单独硬编码 accuracy/balanced_accuracy/pr_auc/roc_auc = 0.0。旧代码在退化时
    压根不调用 sklearn，四个指标全部写死 0.0——这会让 loso_mode 用 val_bacc 选
    best epoch 时，只要某折的 val 受试者是单类别，每个 epoch 都读到同一个硬编码
    0.0，best_epoch 就跟模型实际训练效果无关。这里改成 accuracy/balanced_accuracy
    照常计算（对单类别 y_true 也有明确定义），只有数学上未定义的 roc_auc/pr_auc
    由 binary_metrics_fn 内部返回 NaN——跟 cbramod_finetune/finetune_evaluator.py
    的 get_metrics_for_binaryclass 是同一套处理方式。"""
    return binary_metrics_fn(
        gt, result,
        metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
        threshold=threshold,
    )


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model, test_loader=None):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.test_loader = test_loader
        # [PL 2.x 兼容] validation_epoch_end(self, outputs)/test_epoch_end(self, outputs)
        # 在 pytorch_lightning>=2.0 里被整个移除（不是弃用——Trainer 启动时直接
        # NotImplementedError），官方迁移方式是改用不带参数的
        # on_validation_epoch_end(self)/on_test_epoch_end(self)，自己在 *_step
        # 里把结果攒到实例属性里。跟 run_multiclass_supervised.py 的
        # LitModel_finetune 已经做过的同一个迁移保持一致写法。
        self.val_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        prob = self.model(X)
        loss = BCE(prob, y, self.args.pos_weight)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()

        self.val_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    # [PL 2.x 兼容] validation_epoch_end -> on_validation_epoch_end，见 __init__ 里的说明
    def on_validation_epoch_end(self):
        if len(self.val_step_outputs) == 0:
            return

        result = np.array([])
        gt = np.array([])
        for out in self.val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        self.val_step_outputs.clear()

        if not _binary_degenerate(gt):
            # top-k 自适应阈值：让预测正类数量等于真实正类数量（BIOT 原作者的选阈值方式，
            # LOSO 改动不动它）。
            self.threshold = np.sort(result)[-int(np.sum(gt))]
        else:
            # gt 全 0/全 1 时 int(np.sum(gt)) 是 0 或 len(gt)，np.sort(result)[-0]/[-len(gt)]
            # 都会退化成取最小概率当阈值，跟"让正类预测数=真实正类数"这个公式的原意无关
            # （因为真实正类数是 0 或全部）——退化时固定用 0.5。
            self.threshold = 0.5
        val_result = _compute_binary_metrics(gt, result, self.threshold)
        self.log("val_acc", val_result["accuracy"], sync_dist=True)
        self.log("val_bacc", val_result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", val_result["pr_auc"], sync_dist=True)
        self.log("val_auroc", val_result["roc_auc"], sync_dist=True)

        test_results = self._run_test_epoch()

        if self.logger: # 確保 logger 存在
            log_dir = self.logger.log_dir
            log_file = os.path.join(log_dir, "training_logs.jsonl")
            
            # 建立日誌條目
            log_entry = {
                'epoch': self.current_epoch,
                'step': self.global_step,
                'type': 'validation+test',
                'val_metrics': val_result,
                'test_metrics': test_results,
            }
            
            # 以附加模式 (append) 寫入，確保每次 epoch 都是新的一行
            try:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                print(f"Warning: Could not write to metrics.jsonl: {e}")

    def _run_test_epoch(self):
        """Run one test epoch manually during validation."""
        self.model.eval()
        preds, targets = [], []

        # 取得 test dataloader
        test_loader = self.test_loader
        if test_loader is None:
            print("Warning: No test dataloader found, skipping test evaluation.")
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }

        with torch.no_grad():
            for batch in test_loader:
                X, y = batch
                X = X.to(self.device)
                y = y.to(self.device)

                prob = torch.sigmoid(self.model(X))
                preds.append(prob.cpu().numpy())
                targets.append(y.cpu().numpy())

        # 合併所有 batch 結果
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        # 計算 metrics（沿用 self.threshold，不在这里重新选阈值；退化 batch 也照常
        # 计算 accuracy/balanced_accuracy，只有 roc_auc/pr_auc 在未定义时是 NaN）
        test_result = _compute_binary_metrics(targets, preds, self.threshold)

        # log for monitoring
        self.log("test_acc", test_result["accuracy"], sync_dist=True)
        self.log("test_bacc", test_result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", test_result["pr_auc"], sync_dist=True)
        self.log("test_auroc", test_result["roc_auc"], sync_dist=True)

        self.model.train()  
        return test_result

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)

            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()

        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    # [PL 2.x 兼容] test_epoch_end -> on_test_epoch_end，见 __init__ 里的说明
    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            return

        result = np.array([])
        gt = np.array([])
        for out in self.test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        self.test_step_outputs.clear()
        result = _compute_binary_metrics(gt, result, self.threshold)
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)

        if self.logger: # 確保 logger 存在
            log_dir = self.logger.log_dir
            log_file = os.path.join(log_dir, "training_logs.jsonl")
            
            # 建立日誌條目
            log_entry = {
                'epoch': self.current_epoch,
                'step': self.global_step,
                'type': 'test',
                'metrics': result
            }
            
            # 以附加模式 (append) 寫入
            try:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                print(f"Warning: Could not write to metrics.jsonl: {e}")
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def prepare_KaggleERN_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    dataset_paths = {
        "KaggleERN": "/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n56-biot",
        "KaggleERN-ST": "kaggle_data_st",
    }
    if args.dataset not in dataset_paths:
        raise ValueError(f"Undefined dataset: {args.dataset}")

    root = getattr(args, "dataset_dir", None) or dataset_paths[args.dataset]

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        KaggleERNLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        KaggleERNLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        KaggleERNLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_TUAB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    dataset_paths = {
        "TUAB": "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed",
        "CustomStress-16chan": "/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_16chan_no400up_siwen42",
        "CustomStress-30chan": "/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42",
        }
    if args.dataset not in dataset_paths:
        raise ValueError(f"Undefined dataset: {args.dataset}")

    root = dataset_paths[args.dataset]

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_Stress_dataloader(args):
    """[LOSO] Stress dataset loader，支持 --split_mode random_epoch（旧逻辑，读本地
    已经按 epoch 随机切好的 train/val/test 目录）和 subject_independent（新增，17折
    受试者独立 LOSO）。跟 run_multiclass_supervised.py 的 prepare_Motion_dataloader
    是同一个模式：--dataset Stress 用 BIOT 自己的预处理管线（Stress_data，200Hz，
    无滤波，见 stress_data/_run_biot_stress_preprocess.py 的 preprocess_stress_biot），
    --dataset Stress-ST 用 STTransformer 自己的预处理管线（Stress_data_ST，250Hz，
    4-40Hz 带通，见 stress_data/_run_st_stress_preprocess.py 的
    preprocess_stress_sttransformer）——跟 Motion/Motion-ST 用 AllSubjects_Epochs
    (BIOT, 200Hz) vs Motiondata_ST (ST, 250Hz) 两套目录是同一个道理。旧的
    CustomStress-16chan/CustomStress-30chan（走 prepare_TUAB_dataloader）不受影响。
    """
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    is_st = args.dataset == "Stress-ST"
    root = "Stress_data_ST" if is_st else "Stress_data"
    default_rate = 250 if is_st else 200

    split_mode = getattr(args, "split_mode", "random_epoch")

    if split_mode == "random_epoch":
        train_files = os.listdir(os.path.join(root, "train"))
        np.random.shuffle(train_files)
        val_files = os.listdir(os.path.join(root, "val"))
        test_files = os.listdir(os.path.join(root, "test"))

        print("train/val/test:", len(train_files), len(val_files), len(test_files))

        train_loader = torch.utils.data.DataLoader(
            StressLoader(
                os.path.join(root, "train"), train_files,
                sampling_rate=args.sampling_rate, default_rate=default_rate,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            persistent_workers=False,
        )
        val_loader = torch.utils.data.DataLoader(
            StressLoader(
                os.path.join(root, "val"), val_files,
                sampling_rate=args.sampling_rate, default_rate=default_rate,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )
        test_loader = torch.utils.data.DataLoader(
            StressLoader(
                os.path.join(root, "test"), test_files,
                sampling_rate=args.sampling_rate, default_rate=default_rate,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )
        print("Loader sizes:", len(train_loader), len(val_loader), len(test_loader))
        return train_loader, test_loader, val_loader, None

    elif split_mode == "subject_independent":
        if not args.test_subject or not args.val_subject:
            raise ValueError(
                "split_mode=subject_independent requires --test_subject and --val_subject"
            )
        if args.test_subject == args.val_subject:
            raise ValueError("--test_subject and --val_subject must be different")

        subject_to_files = list_stress_files_by_subject(root)
        subjects = sorted(subject_to_files.keys(), key=lambda s: int(s[3:]))
        for s in (args.test_subject, args.val_subject):
            if s not in subject_to_files:
                raise ValueError(f"Subject {s!r} not found among {subjects}")

        train_subjects = [s for s in subjects if s not in (args.test_subject, args.val_subject)]
        val_subjects = [args.val_subject]
        test_subjects = [args.test_subject]

        def gather(subj_list):
            files = []
            for s in subj_list:
                files.extend(subject_to_files[s])
            return files

        train_files = gather(train_subjects)
        val_files = gather(val_subjects)
        test_files = gather(test_subjects)

        print(f"[split_mode=subject_independent] fold={args.fold_idx}  "
              f"test={args.test_subject}  val={args.val_subject}  "
              f"train={len(train_subjects)} subjects")
        print("train/val/test files:", len(train_files), len(val_files), len(test_files))

        train_loader = torch.utils.data.DataLoader(
            StressLoader(
                "", train_files,
                sampling_rate=args.sampling_rate, default_rate=default_rate,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            persistent_workers=False,
        )
        val_loader = torch.utils.data.DataLoader(
            StressLoader(
                "", val_files,
                sampling_rate=args.sampling_rate, default_rate=default_rate,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )
        test_loader = torch.utils.data.DataLoader(
            StressLoader(
                "", test_files,
                sampling_rate=args.sampling_rate, default_rate=default_rate,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )
        print("Loader sizes:", len(train_loader), len(val_loader), len(test_loader))
        stress_test_meta = {"root": "", "files": test_files}
        return train_loader, test_loader, val_loader, stress_test_meta

    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")


def prepare_CHB_MIT_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_segments"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "train"),
                     train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "test"),
                     test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_PTB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/WFDB/processed2"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


class BestEpochTracker(pl.Callback):
    """[LOSO 新增] 纯观察者：记录验证集 val_bacc 最好的那个 epoch（1-indexed），
    不修改 LitModel_finetune 的任何训练/验证逻辑，只读 trainer.callback_metrics。
    跟 run_multiclass_supervised.py 里 Motion-LOSO 用的是同一个类。"""

    def __init__(self, monitor="val_bacc"):
        self.monitor = monitor
        self.best_score = float("-inf")
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        score = float(score)
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = trainer.current_epoch + 1


def save_stress_fold_results(args, lightning_model, checkpoint_callback, stress_test_meta,
                              best_epoch, train_time_sec, save_dir):
    """[LOSO 新增] 用被选中的最佳 epoch 权重对 test 集重新推理一次，保存
    {task}_{model}_fold{i:02d}.npz (sample_id/y_true/y_pred/y_prob/subject_id) 和
    .json (fold/test_subject/val_subject/balanced_accuracy/best_epoch/hyperparams/
    train_time_sec/peak_gpu_mem_mb/gpu_name)。只在 Stress/Stress-ST +
    subject_independent 时被调用；失败直接抛异常，不静默跳过。

    模型输出是单个 logit（BCEWithLogitsLoss 用，不是多分类的 (batch, n_classes)），
    这里 sigmoid+0.5 阈值，并把 y_prob 拼成 (N, 2) 的 [1-P(positive), P(positive)]，
    跟 cbramod_finetune/finetune_evaluator.py、labram_finetune 的 save_loso_fold_results
    用同一套 npz schema（y_prob.argmax(axis=1)==y_pred 的自洽性检查因此不用改）。
    """
    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        raise RuntimeError(
            f"save_stress_fold_results: no best checkpoint found (best_model_path={best_ckpt_path!r})"
        )
    state = torch.load(best_ckpt_path, map_location="cpu")
    lightning_model.load_state_dict(state["state_dict"])

    model = lightning_model.model
    device = next(model.parameters()).device
    model.eval()

    is_st = args.dataset == "Stress-ST"
    default_rate = 250 if is_st else 200
    sid_test_loader = torch.utils.data.DataLoader(
        StressLoader(
            stress_test_meta["root"], stress_test_meta["files"],
            sampling_rate=args.sampling_rate, default_rate=default_rate,
            return_sample_id=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_stress_with_sample_id,
    )

    subj_re = re.compile(r'^S(\d+)_')
    sample_ids, y_true, y_pred, y_prob, subject_ids = [], [], [], [], []
    with torch.no_grad():
        for X, Y, batch_sample_ids in sid_test_loader:
            X = X.to(device)
            logits = model(X).view(-1)
            prob_pos = torch.sigmoid(logits)
            preds = (prob_pos > 0.5).long()
            probs = torch.stack([1 - prob_pos, prob_pos], dim=-1)
            for i, sid in enumerate(batch_sample_ids):
                m = subj_re.match(sid)
                if not m:
                    raise ValueError(f"Cannot parse subject_id from sample_id: {sid!r}")
                sample_ids.append(sid)
                y_true.append(int(Y[i].item()))
                y_pred.append(int(preds[i].cpu().item()))
                y_prob.append(probs[i].cpu().numpy())
                subject_ids.append(int(m.group(1)))

    if len(sample_ids) == 0:
        raise RuntimeError("save_stress_fold_results: no samples collected, refusing to save an empty file.")

    sample_ids_arr = np.array(sample_ids)
    order = np.argsort(sample_ids_arr)
    sample_ids_arr = sample_ids_arr[order]
    y_true_arr = np.array(y_true, dtype=np.int64)[order]
    y_pred_arr = np.array(y_pred, dtype=np.int64)[order]
    y_prob_arr = np.array(y_prob, dtype=np.float32)[order]
    subject_id_arr = np.array(subject_ids, dtype=np.int64)[order]

    task = args.task_name if args.task_name else args.dataset.lower()
    model_name = args.model_name
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{task}_{model_name}_fold{args.fold_idx:02d}.npz")
    json_path = os.path.join(save_dir, f"{task}_{model_name}_fold{args.fold_idx:02d}.json")

    # 已有旧结果先改名备份，绝不静默覆盖
    for path in (npz_path, json_path):
        if os.path.exists(path):
            ts = _time.strftime("%Y%m%d-%H%M%S")
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
        raise RuntimeError(f"save_stress_fold_results: failed to write {npz_path}")
    _reload = np.load(npz_path)
    for key in ("sample_id", "y_true", "y_pred", "y_prob", "subject_id"):
        if key not in _reload:
            raise RuntimeError(f"save_stress_fold_results: {npz_path} missing key {key!r} after write")

    balanced_accuracy = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    hyperparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "seed": 12345,
        "model": args.model,
        "dataset_channels": args.dataset_channels,
        "in_channels": args.in_channels,
        "sample_length": args.sample_length,
        "sampling_rate": args.sampling_rate,
        "token_size": args.token_size,
        "hop_length": args.hop_length,
        "freeze_backbone": bool(args.freeze_backbone),
    }

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated(0) / (1024 ** 2)) if torch.cuda.is_available() else None

    meta = {
        "fold": args.fold_idx,
        "test_subject": args.test_subject,
        "val_subject": args.val_subject,
        "balanced_accuracy": balanced_accuracy,
        "best_epoch": best_epoch,
        "saved_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hyperparams": hyperparams,
        "train_time_sec": train_time_sec,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "gpu_name": gpu_name,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if not os.path.exists(json_path):
        raise RuntimeError(f"save_stress_fold_results: failed to write {json_path}")
    with open(json_path) as f:
        json.load(f)  # 回读校验 JSON 没写坏

    print(f"Saved fold predictions to {npz_path}")
    print(f"Saved fold metadata to {json_path}")
    print(f"  fold={args.fold_idx} test={args.test_subject} val={args.val_subject} "
          f"best_epoch={best_epoch} balanced_accuracy={balanced_accuracy:.5f}")


def save_kaggleern_epoch_results(args, lightning_model, checkpoint_callback,
                                  best_epoch, train_time_sec, save_dir):
    """[KaggleERN bestval] 用 val_bacc 最好的那个 epoch(checkpoint_callback.best_model_path)
    对 val/test 集各重新做一次干净推理，保存：
      kaggleern_{model}_val.npz / kaggleern_{model}_test.npz
        -- sample_id(=epoch_id，已排序) / y_true / y_pred / y_prob(N,2) / subject_id
      kaggleern_{model}.json
        -- 模型名/任务/超参数/总epoch数/best_epoch/val_bacc/test_bacc/checkpoint路径/
           val与test各类样本数

    跟 run_multiclass_supervised.py 的 save_sleep_epoch_results 同一个思路（KaggleERN
    跟 Sleep 一样是固定的 train/val/test 单次划分，不是 LOSO，所以 val/test 都要存）；
    区别在于这里是二分类，模型输出单个 logit，sigmoid+0.5 阈值出 y_pred，y_prob 拼成
    (N, 2) 的 [1-P(positive), P(positive)]，跟 save_stress_fold_results 用同一套约定。

    这里重新构造 val/test 的 DataLoader（KaggleERNLoader(..., return_sample_id=True)），
    不复用 supervised() 里已经建好的 val_loader/test_loader——那两个是训练用的普通
    (X, Y) 二元组，不带 sample_id。跟 save_stress_fold_results 是同一个思路。
    任何失败都直接抛异常，不静默跳过。
    """
    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        raise RuntimeError(
            f"save_kaggleern_epoch_results: no best checkpoint found (best_model_path={best_ckpt_path!r})"
        )
    state = torch.load(best_ckpt_path, map_location="cpu")
    lightning_model.load_state_dict(state["state_dict"])

    model = lightning_model.model
    device = next(model.parameters()).device
    model.eval()

    dataset_paths = {
        "KaggleERN": "/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n56-biot",
        "KaggleERN-ST": "kaggle_data_st",
    }
    root = getattr(args, "dataset_dir", None) or dataset_paths[args.dataset]

    task = args.task_name if args.task_name else args.dataset.lower()
    model_name = args.model_name
    os.makedirs(save_dir, exist_ok=True)

    subj_re = re.compile(r'^S(\d+)_')

    def _run_split(split_name):
        files = os.listdir(os.path.join(root, split_name))
        loader = torch.utils.data.DataLoader(
            KaggleERNLoader(os.path.join(root, split_name), files,
                             sampling_rate=args.sampling_rate, return_sample_id=True),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_kaggleern_with_sample_id,
        )

        sample_ids, y_true, y_pred, y_prob, subject_ids = [], [], [], [], []
        with torch.no_grad():
            for X, Y, batch_sample_ids in loader:
                X = X.to(device)
                logits = model(X).view(-1)
                prob_pos = torch.sigmoid(logits)
                preds = (prob_pos > 0.5).long()
                probs = torch.stack([1 - prob_pos, prob_pos], dim=-1)
                for i, sid in enumerate(batch_sample_ids):
                    m = subj_re.match(sid)
                    if not m:
                        raise ValueError(f"Cannot parse subject_id from sample_id: {sid!r}")
                    sample_ids.append(sid)
                    y_true.append(int(Y[i].item()))
                    y_pred.append(int(preds[i].cpu().item()))
                    y_prob.append(probs[i].cpu().numpy())
                    subject_ids.append(int(m.group(1)))

        if len(sample_ids) == 0:
            raise RuntimeError(f"save_kaggleern_epoch_results: no samples collected for split={split_name!r}.")

        sample_ids_arr = np.array(sample_ids)
        order = np.argsort(sample_ids_arr)
        sample_ids_arr = sample_ids_arr[order]
        y_true_arr = np.array(y_true, dtype=np.int64)[order]
        y_pred_arr = np.array(y_pred, dtype=np.int64)[order]
        y_prob_arr = np.array(y_prob, dtype=np.float32)[order]
        subject_id_arr = np.array(subject_ids, dtype=np.int64)[order]

        npz_path = os.path.join(save_dir, f"{task}_{model_name}_{split_name}.npz")
        if os.path.exists(npz_path):
            ts = _time.strftime("%Y%m%d-%H%M%S")
            backup_path = f"{npz_path}.bak-{ts}"
            os.rename(npz_path, backup_path)
            print(f"[warn] existing kaggleern result found, backed up to {backup_path}")

        np.savez(
            npz_path,
            sample_id=sample_ids_arr, y_true=y_true_arr, y_pred=y_pred_arr, y_prob=y_prob_arr,
            subject_id=subject_id_arr,
        )
        if not os.path.exists(npz_path):
            raise RuntimeError(f"save_kaggleern_epoch_results: failed to write {npz_path}")
        _reload = np.load(npz_path)
        for key in ("sample_id", "y_true", "y_pred", "y_prob", "subject_id"):
            if key not in _reload:
                raise RuntimeError(f"save_kaggleern_epoch_results: {npz_path} missing key {key!r} after write")

        bacc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        class_counts = np.bincount(y_true_arr, minlength=2).tolist()
        print(f"Saved kaggleern {split_name} predictions to {npz_path} (balanced_accuracy={bacc:.5f})")
        return npz_path, bacc, class_counts

    val_npz_path, val_bacc, val_class_counts = _run_split("val")
    test_npz_path, test_bacc, test_class_counts = _run_split("test")

    hyperparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "model": args.model,
        "dataset_channels": args.dataset_channels,
        "in_channels": args.in_channels,
        "sample_length": args.sample_length,
        "sampling_rate": args.sampling_rate,
        "token_size": args.token_size,
        "hop_length": args.hop_length,
        "freeze_backbone": bool(args.freeze_backbone),
    }

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated(0) / (1024 ** 2)) if torch.cuda.is_available() else None

    meta = {
        "model_name": model_name,
        "task": task,
        "dataset": args.dataset,
        "split_mode": "pooled_random_epoch",  # 见 preprocess_KaggleERN_new.ipynb：全体受试者按 epoch 随机切分，不是 LOSO
        "total_epochs": args.epochs,
        "best_epoch": best_epoch,
        "val_balanced_accuracy": val_bacc,
        "test_balanced_accuracy": test_bacc,
        "val_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "checkpoint_path": best_ckpt_path,
        "val_npz_path": val_npz_path,
        "test_npz_path": test_npz_path,
        "saved_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hyperparams": hyperparams,
        "train_time_sec": train_time_sec,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "gpu_name": gpu_name,
    }

    json_path = os.path.join(save_dir, f"{task}_{model_name}.json")
    if os.path.exists(json_path):
        ts = _time.strftime("%Y%m%d-%H%M%S")
        backup_path = f"{json_path}.bak-{ts}"
        os.rename(json_path, backup_path)
        print(f"[warn] existing kaggleern sidecar json found, backed up to {backup_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if not os.path.exists(json_path):
        raise RuntimeError(f"save_kaggleern_epoch_results: failed to write {json_path}")
    with open(json_path) as f:
        json.load(f)  # 回读校验 JSON 没写坏

    print(f"Saved kaggleern metadata to {json_path}")
    print(f"  best_epoch={best_epoch} val_bacc={val_bacc:.5f} test_bacc={test_bacc:.5f}")


def supervised(args):
    stress_test_meta = None
    is_stress_loso = (
        args.dataset in ("Stress", "Stress-ST")
        and getattr(args, "split_mode", "random_epoch") == "subject_independent"
    )

    # get data loaders
    if args.dataset in ["TUAB", "CustomStress-16chan", "CustomStress-30chan"]:
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)
    elif args.dataset in ["KaggleERN", "KaggleERN-ST"]:
        train_loader, test_loader, val_loader = prepare_KaggleERN_dataloader(args)
    elif args.dataset in ["Stress", "Stress-ST"]:
        train_loader, test_loader, val_loader, stress_test_meta = prepare_Stress_dataloader(args)
    else:
        raise NotImplementedError

    # define the model
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sampling_rate * args.sample_length),
            n_classes=args.n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif args.model == "ContraWR":
        model = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif args.model == "CNNTransformer":
        model = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.sampling_rate,
            steps=args.hop_length // 5,
            dropout=0.2,
            nhead=4,
            emb_size=256,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sampling_rate * args.sample_length),
            shrink_steps=20,
        )

    elif args.model == "STTransformer":
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(
                args.sampling_rate * args.sample_length
            ),  # (sampling_rate * duration)
            n_channels=args.in_channels,
        )

    elif args.model == "BIOT":
        if args.dataset_channels ==  args.in_channels:
            model = BIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        else:
            model = Ada_BIOT(
                input_chan_size=args.dataset_channels,
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    elif args.model == "LabramClassifier-BIOT":
        if args.dataset_channels ==  args.in_channels:
            model = Labram_style_BIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        else:
            model = Labram_style_Ada_BIOT(
                input_chan_size=args.dataset_channels,
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")
    elif args.model == "CBraMod_3lyStyle_LayerNorm-BIOT":
        if args.dataset_channels ==  args.in_channels:
            model = CBraMod_3lyStyle_LayerNorm_BIOT(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        else:
            model = CBraMod_3lyStyle_LayerNorm_Ada_BIOT(
                input_chan_size=args.dataset_channels,
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")
    else:
        raise NotImplementedError

    if args.freeze_backbone:
        print("Freezing parameters for model.biot...")
        for param in model.biot.parameters():
            param.requires_grad = False
        print("Parameters frozen.")

    lightning_model = LitModel_finetune(args, model, test_loader=test_loader)

    # logger and callbacks
    version = f"{args.output_dir}/{args.exp_name}-lr{args.lr}-bs{args.batch_size}-wd{args.weight_decay}-sr{args.sampling_rate}-ts{args.token_size}-hl{args.hop_length}"
    if is_stress_loso:
        # [LOSO 新增] 17折共用同一组超参数，仅 test/val 受试者不同；不加区分会导致
        # 全部折落在同一个 ./log/{version} 目录，互相覆盖 checkpoint/日志。时间戳
        # 保证同一折被重跑（比如中断后重试）也不会覆盖上一次的产物。
        run_timestamp = _time.strftime("%Y%m%d-%H%M%S")
        version += f"-fold{args.fold_idx:02d}-test{args.test_subject}-{run_timestamp}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )

    # 將所有參數 (args) 儲存為 config.yaml
    log_dir = os.path.join("./log", version)
    os.makedirs(log_dir, exist_ok=True)
    args_dict = vars(args)
    args_yaml_path = os.path.join(log_dir, "config.yaml")
    try:
        with open(args_yaml_path, 'w') as f:
            yaml.dump(args_dict, f, sort_keys=False)
        print(f"Configuration saved to {args_yaml_path}")
    except Exception as e:
        print(f"Could not save config.yaml: {e}. (Is 'pyyaml' installed?)")
    # 將模型結構儲存為 model_structure.txt
    model_txt_path = os.path.join(log_dir, "model_structure.txt")
    try:
        with open(model_txt_path, 'w') as f:
            f.write(str(lightning_model.model))
        print(f"Model structure saved to {model_txt_path}")
    except Exception as e:
        print(f"Error saving model_structure.txt: {e}")

    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )

    # [修复] 此前 callbacks=[] 意味着 enable_checkpointing=True 用的是 PL 默认的
    # "最后一轮" checkpoint,下面 trainer.test(ckpt_path="best") 名不副实地一直在
    # 用最后一轮而不是验证集 val_bacc 最好的一轮。这里补上 ModelCheckpoint(monitor="val_bacc"),
    # 和 run_multiclass_supervised.py 里 Motion-LOSO 路径已经验证过的写法保持一致。
    checkpoint_callback = ModelCheckpoint(
        monitor="val_bacc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_bacc:.5f}",
        auto_insert_metric_name=False,
    )
    best_epoch_tracker = BestEpochTracker(monitor="val_bacc")
    callbacks = [checkpoint_callback, best_epoch_tracker]

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        # strategy=DDPStrategy(find_unused_parameters=False),
        # auto_select_gpus was removed in pytorch_lightning >=2.0 (TypeError:
        # unexpected keyword argument); devices=[0] combined with the launcher
        # scripts' CUDA_VISIBLE_DEVICES already pins the exact GPU, so this
        # was never doing anything auto_select_gpus=False wouldn't already do.
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        # callbacks=[early_stop_callback],
    )

    # train the model
    is_kaggleern = args.dataset in ("KaggleERN", "KaggleERN-ST")
    if (is_stress_loso or is_kaggleern) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start_time = _time.time()
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    train_time_sec = _time.time() - train_start_time

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(pretrain_result)

    # [LOSO 新增] 用最佳 epoch 权重对 test 集重新推理，落盘
    # sample_id/y_true/y_pred/y_prob/subject_id 供事后复现所有指标
    if is_stress_loso:
        save_stress_fold_results(
            args, lightning_model, checkpoint_callback, stress_test_meta,
            best_epoch_tracker.best_epoch, train_time_sec, save_dir=(args.fold_results_dir or log_dir),
        )

    # [KaggleERN bestval] KaggleERN 是固定 train/val/test 单次划分（不是 LOSO），
    # 跟 Sleep 一样 val/test 都要存，用 val_bacc 最好的 epoch 重新推理两遍。
    if is_kaggleern:
        save_kaggleern_epoch_results(
            args, lightning_model, checkpoint_callback,
            best_epoch_tracker.best_epoch, train_time_sec, save_dir=(args.fold_results_dir or log_dir),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="finetune", help="experiment name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--dataset_channels", type=int, default=None,  # 如果為 None，則使用 in_channels
        help="actual number of channels in dataset (if different from pretrained model)"
    )
    parser.add_argument('--pos_weight', default=None, type=float)
    parser.add_argument(
        "--model", type=str, default="SPaRCNet", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=16, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./", help="saved model path"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the backbone during training (default: False)."
    )
    # ===== [LOSO 新增] Stress 17折 subject-independent LOSO 相关参数 =====
    # 默认值完全复现今天的行为：split_mode 默认 random_epoch，其余数据集/
    # Stress 的旧路径（CustomStress-16chan/CustomStress-30chan）都不读这些新参数。
    parser.add_argument(
        "--split_mode", type=str, default="random_epoch",
        choices=["random_epoch", "subject_independent"],
        help="Stress dataset split strategy; only --dataset Stress/Stress-ST reads this."
    )
    parser.add_argument("--test_subject", type=str, default=None,
                        help="e.g. Sub04; required when split_mode=subject_independent")
    parser.add_argument("--val_subject", type=str, default=None,
                        help="e.g. Sub05; required when split_mode=subject_independent")
    parser.add_argument("--fold_idx", type=int, default=0,
                        help="LOSO fold index (0-based), used in saved filenames/log dir")
    parser.add_argument("--model_name", type=str, default="biot",
                        help="model label used in saved {task}_{model}_fold{i}.npz/json filenames")
    parser.add_argument("--task_name", type=str, default=None,
                        help="task label used in saved filenames; defaults to dataset.lower()")
    parser.add_argument("--fold_results_dir", type=str, default=None,
                        help="where to save {task}_{model}_fold{i}.npz/json; defaults to the run's log_dir")
    # [KaggleERN bestval] 覆盖 prepare_KaggleERN_dataloader() 里硬编码的远程绝对路径；
    # 不传时行为完全不变（还是走 dataset_paths["KaggleERN"] 那个绝对路径），只有显式传
    # --dataset_dir 才会用这里的目录（smoke 测试时指向 ./biot_kaggleern_data_smoke）。
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Override the dataset root directory for --dataset KaggleERN "
                             "(must contain train/val/test subdirs); default uses the hardcoded path.")
    args = parser.parse_args()

    if args.dataset_channels is None:
        args.dataset_channels = args.in_channels

    print(args)

    supervised(args)
