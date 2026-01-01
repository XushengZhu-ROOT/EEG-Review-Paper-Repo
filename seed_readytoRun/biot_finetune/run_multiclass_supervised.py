import os
import argparse
import pickle

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
    confusion_matrix, 
    balanced_accuracy_score, 
    classification_report,
    accuracy_score,
    f1_score,
    cohen_kappa_score
)


def multiclass_metrics_fn(y_true, y_pred_proba, metrics=None):
    """
    使用 sklearn 实现多分类指标计算
    替换 pyhealth.metrics.multiclass_metrics_fn
    
    Args:
        y_true: 真实标签 (1D array)
        y_pred_proba: 预测概率 (2D array, shape: [n_samples, n_classes])
        metrics: 要计算的指标列表
    
    Returns:
        dict: 包含各项指标的字典
    """
    if metrics is None:
        metrics = ["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"]
    
    # 获取预测类别
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    result = {}
    
    if "accuracy" in metrics:
        result["accuracy"] = accuracy_score(y_true, y_pred)
    
    if "f1_macro" in metrics:
        result["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    if "f1_weighted" in metrics:
        result["f1_weighted"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if "cohen_kappa" in metrics:
        result["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    
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
    CBraMod_3lyStyle_LayerNorm_Ada_BIOT,
)
from utils import TUEVLoader, HARLoader, MotionLoader, SEEDLoader

import json
import yaml

class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model, test_loader=None):
        super().__init__()
        self.model = model
        self.args = args
        self.test_loader = test_loader
        self.criterion = torch.nn.CrossEntropyLoss()

        # ---- NEW: buffers for epoch-end hooks in PL v2 ----
        self.val_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)              # shape: (B, n_classes)
        loss = self.criterion(logits, y.long())  # y shape: (B,)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        # Lightning 會自動在 val loop 關閉 grad，其實不用 no_grad，但保留也沒關係
        with torch.no_grad():
            logits = self.model(X)
            prob = torch.softmax(logits, dim=1)
            step_result = prob.detach().cpu().numpy()
            step_gt = y.detach().cpu().numpy()

        # ---- NEW: save outputs to buffer, do not rely on outputs param ----
        self.val_step_outputs.append((step_result, step_gt))
        # 這邊不用 return 也可以，但保留 return 也不會錯
        return step_result, step_gt

    # ---- REPLACED: validation_epoch_end -> on_validation_epoch_end ----
    def on_validation_epoch_end(self):
        if len(self.val_step_outputs) == 0:
            return

        preds = []
        targets = []

        for pred, tgt in self.val_step_outputs:
            preds.append(pred)   # pred shape: (B, n_classes)
            targets.append(tgt)  # tgt  shape: (B,)

        preds = np.concatenate(preds, axis=0)      # (N, n_classes)
        targets = np.concatenate(targets, axis=0)  # (N,)

        # 清 buffer，避免 memory 疊加
        self.val_step_outputs.clear()

        # 获取预测类别
        pred_classes = np.argmax(preds, axis=1)
        
        # 计算更多评估指标
        val_result = multiclass_metrics_fn(
            targets,
            preds,
            metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
        )
        
        # 添加balanced_accuracy
        val_result["balanced_accuracy"] = balanced_accuracy_score(targets, pred_classes)
        
        # 计算混淆矩阵
        cm = confusion_matrix(targets, pred_classes)
        val_result["confusion_matrix"] = cm.tolist()
        
        self.log("val_acc", val_result["accuracy"], sync_dist=True, prog_bar=True)
        self.log("val_f1_macro", val_result["f1_macro"], sync_dist=True, prog_bar=True)
        self.log("val_f1_weighted", val_result["f1_weighted"], sync_dist=True)
        self.log("val_cohen_kappa", val_result["cohen_kappa"], sync_dist=True)
        self.log("val_bacc", val_result["balanced_accuracy"], sync_dist=True, prog_bar=True)

        # 在每個 val epoch 結束跑一次 test（你原本的邏輯）
        test_results = self._run_test_epoch()

        # 保存验证集的混淆矩阵
        if self.logger:
            log_dir = self.logger.log_dir
            val_cm_file = os.path.join(log_dir, f"val_confusion_matrix_epoch_{self.current_epoch}.npy")
            try:
                val_cm = np.array(val_result["confusion_matrix"])
                np.save(val_cm_file, val_cm)
            except Exception as e:
                print(f"Warning: Could not save validation confusion matrix: {e}")
        
        # 把 val + test 的結果一起寫到 jsonl
        if self.logger:  # 確保 logger 存在
            log_dir = self.logger.log_dir
            log_file = os.path.join(log_dir, "training_logs.jsonl")

            log_entry = {
                "epoch": int(self.current_epoch),
                "step": int(self.global_step),
                "type": "validation+test",
                "val_metrics": val_result,
                "test_metrics": test_results,
            }

            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Warning: Could not write to training_logs.jsonl: {e}")

    def _run_test_epoch(self):
        """Run one test epoch manually during validation."""
        self.model.eval()
        preds, targets = [], []

        test_loader = self.test_loader
        if test_loader is None:
            print("Warning: No test dataloader found, skipping test evaluation.")
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "cohen_kappa": 0.0,
                "balanced_accuracy": 0.0,
                "confusion_matrix": [],
            }

        with torch.no_grad():
            for batch in test_loader:
                X, y = batch
                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X)
                prob = torch.softmax(logits, dim=1)
                preds.append(prob.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        # 获取预测类别
        pred_classes = np.argmax(preds, axis=1)
        
        # 计算更多评估指标
        test_result = multiclass_metrics_fn(
            targets,
            preds,
            metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
        )
        
        # 添加balanced_accuracy
        test_result["balanced_accuracy"] = balanced_accuracy_score(targets, pred_classes)
        
        # 计算混淆矩阵
        cm = confusion_matrix(targets, pred_classes)
        test_result["confusion_matrix"] = cm.tolist()

        self.log("test_acc", test_result["accuracy"], sync_dist=True)
        self.log("test_f1_macro", test_result["f1_macro"], sync_dist=True)
        self.log("test_f1_weighted", test_result["f1_weighted"], sync_dist=True)
        self.log("test_cohen_kappa", test_result["cohen_kappa"], sync_dist=True)
        self.log("test_bacc", test_result["balanced_accuracy"], sync_dist=True)

        self.model.train()
        return test_result

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self.model(X)
            prob = torch.softmax(logits, dim=1)
            step_result = prob.detach().cpu().numpy()
            step_gt = y.detach().cpu().numpy()

        # ---- NEW: buffer for on_test_epoch_end ----
        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    # ---- REPLACED: test_epoch_end -> on_test_epoch_end ----
    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            return

        preds = []
        targets = []

        for pred, tgt in self.test_step_outputs:
            preds.append(pred)   # (B, n_classes)
            targets.append(tgt)  # (B,)

        preds = np.concatenate(preds, axis=0)      # (N, n_classes)
        targets = np.concatenate(targets, axis=0)  # (N,)

        # 清 buffer
        self.test_step_outputs.clear()

        # 获取预测类别
        pred_classes = np.argmax(preds, axis=1)
        
        # 计算更多评估指标
        test_result = multiclass_metrics_fn(
            targets,
            preds,
            metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
        )
        
        # 添加balanced_accuracy
        test_result["balanced_accuracy"] = balanced_accuracy_score(targets, pred_classes)
        
        # 计算混淆矩阵
        cm = confusion_matrix(targets, pred_classes)
        test_result["confusion_matrix"] = cm.tolist()
        
        # 计算分类报告（包含per-class指标）
        try:
            class_report = classification_report(
                targets, pred_classes, 
                output_dict=True, 
                zero_division=0
            )
            test_result["classification_report"] = class_report
        except Exception as e:
            print(f"Warning: Could not compute classification report: {e}")
            test_result["classification_report"] = {}

        self.log("test_acc",      test_result["accuracy"],   sync_dist=True)
        self.log("test_f1_macro", test_result["f1_macro"],   sync_dist=True)
        self.log("test_f1_weighted", test_result["f1_weighted"], sync_dist=True)
        self.log("test_cohen_kappa", test_result["cohen_kappa"], sync_dist=True)
        self.log("test_bacc", test_result["balanced_accuracy"], sync_dist=True)

        # 打印混淆矩阵
        print("\n" + "="*50)
        print("混淆矩阵 (Confusion Matrix):")
        print("="*50)
        print(cm)
        print("="*50 + "\n")

        if self.logger:
            log_dir = self.logger.log_dir
            log_file = os.path.join(log_dir, "training_logs.jsonl")
            
            # 保存混淆矩阵为numpy文件（便于后续可视化）
            cm_file = os.path.join(log_dir, f"confusion_matrix_epoch_{self.current_epoch}.npy")
            try:
                np.save(cm_file, cm)
            except Exception as e:
                print(f"Warning: Could not save confusion matrix: {e}")
            
            log_entry = {
                "epoch": int(self.current_epoch),
                "step": int(self.global_step),
                "type": "test",
                "metrics": test_result,
            }
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Warning: Could not write to training_logs.jsonl: {e}")

        return test_result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        return optimizer

def prepare_TUEV_dataloader(args):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_HAR_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/HAR/processed/"

    train_files = os.listdir(os.path.join(root, "train"))
    test_files = os.listdir(os.path.join(root, "test"))
    val_files = os.listdir(os.path.join(root, "val"))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_Motion_dataloader(args):
    # === 固定 random seed ===
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # === 你的 dataset 根目錄 ===
    root = "../../AllSubjects_Epochs"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print("train/val/test:", len(train_files), len(val_files), len(test_files))

    # === DataLoaders ===
    train_loader = torch.utils.data.DataLoader(
        MotionLoader(
            os.path.join(root, "train"),
            train_files,
            sampling_rate=args.sampling_rate,
            in_channels=args.in_channels
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=args.num_workers,
        num_workers=0,
        persistent_workers=False,
    )

    val_loader = torch.utils.data.DataLoader(
        MotionLoader(
            os.path.join(root, "val"),
            val_files,
            sampling_rate=args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
        num_workers=0,
        persistent_workers=False,
    )

    test_loader = torch.utils.data.DataLoader(
        MotionLoader(
            os.path.join(root, "test"),
            test_files,
            sampling_rate=args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
        num_workers=0,
        persistent_workers=False,
    )

    print("Loader sizes:", len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_SEED_dataloader(args):
    """
    准备SEED数据集的数据加载器
    
    根据process_seed.ipynb的预处理：
    - 数据已经resample到200Hz (BIOT) 或 250Hz (STTransformer)
    - label已经是0-6，不需要转换
    - 数据已经预处理过，只需要95%分位数归一化
    
    数据结构：
    - biot_seed_data: train/subject_5/, val/subject_6/, test/subject_4/
    - st_seed_data: train/subject_4/, val/subject_5/, test/subject_6/
    两个数据集都已经按照train/val/test划分好了
    """
    # === 固定 random seed ===
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # === 数据集根目录 ===
    # 根据args.dataset判断使用哪个数据文件夹
    if args.dataset == "SEED":
        # 使用biot_seed_data (BIOT预处理，200Hz)
        root = "biot_seed_data"
    elif args.dataset == "SEED-ST":
        # 使用st_seed_data (STTransformer预处理，250Hz)
        root = "st_seed_data"
    else:
        raise ValueError(f"Unknown SEED dataset: {args.dataset}")

    # === 获取文件列表 ===
    # 两个数据集都已经按照train/val/test划分好了，结构相同
    train_files = []
    train_dir = os.path.join(root, "train")
    if os.path.exists(train_dir):
        for subj_dir in os.listdir(train_dir):
            subj_path = os.path.join(train_dir, subj_dir)
            if os.path.isdir(subj_path):
                subj_files = os.listdir(subj_path)
                # 只保留pickle文件
                subj_files = [f for f in subj_files if f.endswith('.pickle')]
                train_files.extend([os.path.join("train", subj_dir, f) for f in subj_files])
    
    val_files = []
    val_dir = os.path.join(root, "val")
    if os.path.exists(val_dir):
        for subj_dir in os.listdir(val_dir):
            subj_path = os.path.join(val_dir, subj_dir)
            if os.path.isdir(subj_path):
                subj_files = os.listdir(subj_path)
                # 只保留pickle文件
                subj_files = [f for f in subj_files if f.endswith('.pickle')]
                val_files.extend([os.path.join("val", subj_dir, f) for f in subj_files])
    
    test_files = []
    test_dir = os.path.join(root, "test")
    if os.path.exists(test_dir):
        for subj_dir in os.listdir(test_dir):
            subj_path = os.path.join(test_dir, subj_dir)
            if os.path.isdir(subj_path):
                subj_files = os.listdir(subj_path)
                # 只保留pickle文件
                subj_files = [f for f in subj_files if f.endswith('.pickle')]
                test_files.extend([os.path.join("test", subj_dir, f) for f in subj_files])

    np.random.shuffle(train_files)
    print("train/val/test files:", len(train_files), len(val_files), len(test_files))

    # === DataLoaders ===
    # 数据已经预处理过，不需要label_offset和重采样
    train_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            root,
            train_files,
            sampling_rate=args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            root,
            val_files,
            sampling_rate=args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    test_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            root,
            test_files,
            sampling_rate=args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    print("Loader sizes:", len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def supervised(args):
    # get data loaders
    if args.dataset == "TUEV":
        train_loader, test_loader, val_loader = prepare_TUEV_dataloader(args)

    elif args.dataset == "Motion":
        train_loader, test_loader, val_loader = prepare_Motion_dataloader(args)

    elif args.dataset in ["SEED", "SEED-ST"]:
        train_loader, test_loader, val_loader = prepare_SEED_dataloader(args)

    else:
        raise NotImplementedError

    # define the model
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sample_length * args.sampling_rate),
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
            n_segments=4 if args.dataset == "HAR" else 5,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sample_length * args.sampling_rate),
            shrink_steps=16 if args.dataset == "HAR" else 20,
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
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
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

    # 取消早停机制
    # early_stop_callback = EarlyStopping(
    #     monitor="val_f1_macro", patience=5, verbose=True, mode="max"
    # )

    trainer = pl.Trainer(
        # devices=[0],
        accelerator="gpu",
        # strategy=DDPStrategy(find_unused_parameters=False),
        strategy="auto",
        devices=1,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[],  # 移除早停回调
    )
    # trainer = pl.Trainer(
    #     accelerator="cpu",
    #     devices=1,
    #     precision=32,
    #     logger=logger,
    #     max_epochs=args.epochs,
    #     callbacks=[early_stop_callback],
    #     num_sanity_val_steps=0,
    # )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print("\n" + "="*50)
    print("最终测试结果:")
    print("="*50)
    print(pretrain_result)
    
    # 如果结果中包含混淆矩阵，也打印出来
    if "confusion_matrix" in pretrain_result:
        print("\n" + "="*50)
        print("测试集混淆矩阵:")
        print("="*50)
        cm = np.array(pretrain_result["confusion_matrix"])
        print(cm)
        print("="*50 + "\n")


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
        "--in_channels", type=int, default=12, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=6, help="number of output classes"
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
    args = parser.parse_args()
    print(args)

    supervised(args)
