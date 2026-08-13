import random
import os
import math
import re
import time
import argparse
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

from functools import partial
import numpy as np
import random
import os
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from sklearn import metrics
from utils_eval import get_metrics
from einops.layers.torch import Rearrange
import glob
import pickle



def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")



# ==================== 数据相关配置 ====================
# Motor 任务：
# - 六分类（0,1,2,3,4,5）
# - 1 秒，采样率 256 Hz -> 时间长度 256
# - 原始 20 通道，只使用前 19 个通道（丢弃最后一个A2）

use_channels_names = [
    'F7','FP1','FP2','F8','F3','FZ','F4','C3','CZ','P8',
    'P7','PZ','P4','T7','P3','O1','O2','C4','T8'  # 19个通道
]
# 注意：使用前19个通道，丢弃最后一个A2通道

# ===== 跨模型稳定的 sample_id，与 cbramod_finetune/datasets/motortask_dataset.py 的
# compute_sample_id / biot_finetune/utils.py 的 compute_motion_sample_id 保持同步。
# epoch_id 形如 "Sub04_Walkslow_epoch009"；纯函数，只依赖字符串本身，
# 与 shuffle / batch_size / num_workers 无关。三个模型跑同一批底层 epoch 文件
# （cbramod/biot 是 200Hz、eegpt 是 256Hz 重采样，但 epoch_id 完全相同）时，
# 算出来的 sample_id 必须完全一致。 =====
_SAMPLE_ID_TASK_ORDER = ['Walk', '8', 'Horizontal', 'Vertical', 'Pick', 'Stair']
_SAMPLE_ID_SPEED_ORDER = ['slow', 'medium', 'fast']
_SAMPLE_ID_TASK_OFFSET = 3000
_SAMPLE_ID_SPEED_OFFSET = 1000
_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(.+?)_epoch(\d+)$')
_MOTION_SUBJECT_RE = re.compile(r'(Sub\d+)_')


def _parse_task_token(task_token):
    for speed in _SAMPLE_ID_SPEED_ORDER:
        if task_token.endswith(speed):
            return task_token[: -len(speed)], speed
    raise ValueError(f"Cannot parse speed suffix (slow/medium/fast) from task token: {task_token!r}")


def compute_sample_id(epoch_id):
    """由 epoch_id（如 'Sub04_Walkslow_epoch009'）确定性地生成 sample_id，
    格式：S{subject:02d}_ep{index:05d}。"""
    m = _SAMPLE_ID_RE.match(epoch_id)
    if not m:
        raise ValueError(f"Cannot parse epoch_id for sample_id: {epoch_id!r}")
    subject_num = int(m.group(1))
    task_token = m.group(2)
    local_idx = int(m.group(3))
    base_task, speed = _parse_task_token(task_token)
    if base_task not in _SAMPLE_ID_TASK_ORDER:
        raise ValueError(
            f"Unknown base task {base_task!r} parsed from epoch_id {epoch_id!r}; "
            f"expected one of {_SAMPLE_ID_TASK_ORDER}"
        )
    task_idx = _SAMPLE_ID_TASK_ORDER.index(base_task)
    speed_idx = _SAMPLE_ID_SPEED_ORDER.index(speed)
    global_index = task_idx * _SAMPLE_ID_TASK_OFFSET + speed_idx * _SAMPLE_ID_SPEED_OFFSET + local_idx
    if global_index > 99999:
        raise ValueError(f"sample_id index overflow (>99999) for epoch_id {epoch_id!r}: {global_index}")
    return f"S{subject_num:02d}_ep{global_index:05d}"


def extract_motion_subject_id(name):
    """'Sub04_8fast_epoch001.pickle'（或包含它的任意路径）-> 'Sub04'。"""
    m = _MOTION_SUBJECT_RE.search(os.path.basename(name))
    return m.group(1) if m else None


def list_motion_files_by_subject(root):
    """扫描 root/{train,val,test}/*.pickle，按受试者分组（'Sub04' -> [path, ...]），
    用于构造受试者独立（LOSO）划分。"""
    subject_to_files = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".pickle"):
                continue
            sid = extract_motion_subject_id(fname)
            if sid is None:
                continue
            subject_to_files[sid].append(os.path.join(split_dir, fname))
    return subject_to_files


def collate_fn_motion_with_sample_id(batch):
    """配合 KaggleEEGDataset(return_sample_id=True) 使用：batch 是 (X, Y, sample_id) 三元组。"""
    X_list, Y_list, sample_id_list = zip(*batch)
    X_batch = torch.stack(X_list, dim=0)
    Y_batch = torch.stack(Y_list, dim=0)
    return X_batch, Y_batch, list(sample_id_list)


# ==================== 用户可配置参数 ====================
# 建议你只改这一段就能完成大部分自定义

# 任务名称（会用于输出文件夹命名）
TASK_NAME = "EEGPT_Motor"

# 数据路径（Motor 数据目录位置，结构为 Motiondata/{train,val,test}）
data_root = "./Motiondata"

# 分类类别数（Motor: 0~5 六分类）
num_classes = 6

# 训练超参数（默认值，单次跑 main() 会用到）
batch_size = 64
max_epochs = 50
max_lr = 4e-4
weight_decay = 1e-2  # 权重衰减（用于 AdamW）

# 模型微调方式（默认值）
freeze_encoder = False      # True = 线性探测（只训头部），False = 全局微调
encoder_lr_ratio = 0.1      # 全微调时 encoder 学习率 = max_lr * encoder_lr_ratio

# DataLoader 线程数
num_workers = 4

# GPU 配置
# - 如果只用单卡：devices = [0]
# - 如果用多卡：  devices = [0, 1]  之类
# - 如果想让 Lightning 自动选择：可以设为 None
accelerator = "cuda"
devices = [0]   # 根据你机器的 GPU id 修改

# ====== 超参数搜索范围（可选，仅 split_mode=random_epoch 的旧网格搜索流程使用）======
BS_LIST = [128]#[128, 64, 32]
LR_LIST = [1e-3, 4e-4, 1e-4]
WD_LIST = [1e-2, 1e-3]

# ====== LOSO（20折 subject-independent）固定超参数：不做网格搜索 ======
LOSO_BATCH_SIZE = 32
LOSO_LR = 1e-3
LOSO_WEIGHT_DECAY = 1e-2
LOSO_EPOCHS = 50
LOSO_FREEZE_ENCODER = False       # False = 全局微调（full finetune），不是 linear probe
LOSO_ENCODER_LR_RATIO = encoder_lr_ratio


class KaggleEEGDataset(torch.utils.data.Dataset):
    """
    从 motor_data/{split}/*.pickle 读取数据。
    每个 pickle 是一个 dict:
        - 'signal': ndarray, shape (20, 256)
        - 'label' : int, 0~5
        - 'epoch_id': str
    """
    def __init__(self, root_dir: str, split: str = "train", expected_channels: int = 20, target_channels: int = 19,
                 file_list=None, return_sample_id: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.expected_channels = expected_channels
        self.target_channels = target_channels
        self.return_sample_id = return_sample_id

        # ===== [LOSO 新增] 若传入 file_list，直接用该文件列表（用于受试者独立划分），
        # 忽略 root_dir/split 目录扫描；旧调用方式（file_list=None）行为完全不变。=====
        if file_list is not None:
            all_files = list(file_list)
        else:
            self.split_dir = os.path.join(root_dir, split)
            assert os.path.isdir(self.split_dir), f"{self.split_dir} 不存在"
            all_files = [
                os.path.join(self.split_dir, f)
                for f in os.listdir(self.split_dir)
                if f.endswith(".pickle")
            ]

        # 过滤：只保留expected_channels通道的文件
        self.files = []
        skipped = 0
        for f in all_files:
            try:
                with open(f, "rb") as f_obj:
                    obj = pickle.load(f_obj)
                X = np.asarray(obj["signal"], dtype=np.float32)
                if X.shape[0] == expected_channels:
                    self.files.append(f)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
                continue

        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f"split={split}: 没有找到任何 {expected_channels} 通道的 .pickle 文件")

        if skipped > 0:
            print(f"[KaggleEEGDataset] split={split}, 跳过 {skipped} 个通道数不匹配的文件")
        print(f"[KaggleEEGDataset] split={split}, 样本数={len(self.files)}")

        # ===== [LOSO 新增] sample_id 在数据集构造（列文件）时一次性确定，
        # 与 self.files 一一对应；不受 shuffle / batch_size / num_workers 影响。
        # 解析失败直接报错退出（不静默跳过）。=====
        self.sample_ids = [
            compute_sample_id(os.path.splitext(os.path.basename(f))[0]) for f in self.files
        ]
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(f"[{split}] Duplicate sample_id detected after computing sample_ids; "
                              f"check for duplicate/conflicting epoch files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        sample_id = self.sample_ids[idx]
        with open(path, "rb") as f:
            obj = pickle.load(f)

        x = obj["signal"]   # (20, 256) 原始 20 通道
        y = obj["label"]    # int 0~5

        # 检查通道数
        if x.shape[0] != self.expected_channels:
            raise ValueError(f"Expected {self.expected_channels} channels, got {x.shape[0]}")

        # 丢弃最后一个通道(A2)，保留前19个通道
        x = x[:self.target_channels, :]  # (19, 256)

        # 数据清理：NaN/Inf处理
        x = np.nan_to_num(x, posinf=0.0, neginf=0.0)

        x = torch.tensor(x, dtype=torch.float32)  # (C, T)
        y = torch.tensor(y, dtype=torch.long)

        if self.return_sample_id:
            return x, y, sample_id
        return x, y


class LitEEGPTCausal(pl.LightningModule):

    def __init__(
        self,
        load_path: str = "eegpt_mcae_58chs_4s_large4E.ckpt",
        freeze_encoder: bool = True,
        encoder_lr_ratio: float = 0.1,
    ):
        """
        freeze_encoder=True  -> 线性探测（只训练头部）
        freeze_encoder=False -> 全局微调（encoder + 头部）
        """
        super().__init__()
        # Debug 开关：需要时可以改成 False 关闭打印
        self.debug = True
        self.chans_num = len(use_channels_names)  # 19
        self.num_class = num_classes
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_ratio = encoder_lr_ratio  # encoder 学习率相对于 head 的比例
        # test dataloader（在 main 里赋值，用于每个 epoch 做 test）
        self.test_loader = None

        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 1 * 256],  # 19 通道, 1s@256Hz = 256
            patch_size=32*2,  # 64
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        if self.debug:
            print("[Debug] target_encoder.num_patches =", self.target_encoder.num_patches,
                  "type:", type(self.target_encoder.num_patches))

        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
        
        self.target_encoder.load_state_dict(target_encoder_stat)

        # 线性 probe 时冻结 encoder 参数
        if self.freeze_encoder:
            for p in self.target_encoder.parameters():
                p.requires_grad = False

        # 输入为 17 通道（与你的数据一致，如不同需修改这里）
        self.chan_conv       = Conv1dWithConstraint(19, self.chans_num, 1, max_norm=1)
        
        # 分类头结构（按之前的实现：直接展平，不先池化）
        # z 的 shape: (B, N, embed_num, embed_dim) = (B, N, 4, 512)
        # 计算patch数量N
        # patch_size=64, patch_stride=None (默认等于patch_size), seq_len=256
        # 如果patch_stride=None，N = seq_len // patch_size = 256 // 64 = 4
        # 从target_encoder获取num_patches，它是一个tuple (C, N)
        if hasattr(target_encoder, 'num_patches') and isinstance(target_encoder.num_patches, tuple):
            # num_patches是(C, N)格式，C是通道维度的patch数，N是时间维度的patch数
            self.N = target_encoder.num_patches[1]  # N是时间维度的patch数
            if self.debug:
                print(f"[Debug] 从target_encoder获取N={self.N}, num_patches={target_encoder.num_patches}")
        else:
            # 默认计算：假设patch_stride=None，seq_len=256, patch_size=64
            seq_len = 256
            patch_size = 64
            self.N = seq_len // patch_size  # 256 // 64 = 4
            if self.debug:
                print(f"[Debug] 使用默认计算N={self.N} (seq_len={seq_len}, patch_size={patch_size})")
        
        embed_num = 4
        embed_dim = 512
        
        # 分类头：直接展平所有维度，然后通过Linear层
        # Layer 1: (N * embed_num * embed_dim) -> (N * embed_dim)
        # Layer 2: (N * embed_dim) -> embed_dim
        # Layer 3: embed_dim -> num_classes
        self.classifier = nn.Sequential(
            # Reshape: (B, N, embed_num, embed_dim) -> (B, N * embed_num * embed_dim)
            Rearrange('b n e d -> b (n e d)'),
            # Layer 1: (N * embed_num * embed_dim) -> (N * embed_dim)
            nn.Linear(self.N * embed_num * embed_dim, self.N * embed_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            # Layer 2: (N * embed_dim) -> embed_dim
            nn.Linear(self.N * embed_dim, embed_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            # Layer 3: embed_dim -> num_classes
            nn.Linear(embed_dim, self.num_class),
        )
       
        self.drop           = torch.nn.Dropout(p=0.50)
        
        # 损失函数（类别权重将在训练时设置）
        self.loss_fn        = None  # 将在训练时根据类别权重创建
        self.class_weights  = None  # 类别权重
        
        self.running_scores = {"train":[], "valid":[]}
        self.is_sanity=True
        
        # 用于记录混淆矩阵和指标
        self.output_dir = None  # 将在训练时设置
        
    
    def forward(self, x):
        B, C, T = x.shape
        x = x/100
        x = x - x.mean(dim=-2, keepdim=True)
        # Motor 数据为 1s(256)，这里统一插值到 256
        x = temporal_interpolation(x, 1*256)
        x = self.chan_conv(x)
        # 线性 probe 时保持 encoder 在 eval 模式；全微调时交给 Lightning 控制
        if self.freeze_encoder:
            self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))
        
        # z 的 shape: (B, N, embed_num, embed_dim) = (B, N, 4, 512)
        # 直接传入classifier，内部会使用Rearrange展平
        h = self.classifier(z)
        
        return x, h

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        # 如果loss_fn是None，创建默认的CrossEntropyLoss
        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()

        # 多分类：保存整条概率向量用于 epoch 结束时计算指标
        probs = torch.softmax(logit.detach().cpu(), dim=-1)  # (B, num_classes)
        self.running_scores["train"].append((label.clone().detach().cpu(), probs))

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)        # (N,)
        y_score = torch.cat(y_score, dim=0)    # (N, num_classes)
        
        # 多分类预测类别（用于混淆矩阵）
        y_pred = torch.argmax(y_score, dim=-1).numpy()
        label_np = label.numpy()
        
        # 计算多分类指标（使用 sklearn，避免 pyhealth 的 binary/multiclass 限制）
        results = {
            "accuracy": float(accuracy_score(label_np, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(label_np, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(label_np, y_pred)),
            "f1_macro": float(f1_score(label_np, y_pred, average="macro")),
            "f1_weighted": float(f1_score(label_np, y_pred, average="weighted")),
            "f1_micro": float(f1_score(label_np, y_pred, average="micro")),
        }
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(label_np, y_pred)
        
        # 记录指标
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        
        # 打印和保存
        epoch = self.current_epoch
        acc = results.get('accuracy', 0)
        bacc = results.get('balanced_accuracy', 0)
        
        print(f"\n[Epoch {epoch}] Validation - Acc: {acc:.4f}, BAcc: {bacc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        # 保存到文件
        if self.output_dir is not None:
            epoch_file = self.output_dir / f"epoch_{epoch}_valid.json"
            epoch_data = {
                "epoch": epoch,
                "metrics": results,
                "confusion_matrix": cm.tolist(),
            }
            with open(epoch_file, 'w') as f:
                json.dump(epoch_data, f, indent=2)

        # ---------- 每个 epoch 额外在 test 集上评估（如果提供了 test_loader） ----------
        if self.test_loader is not None and self.output_dir is not None:
            all_labels, all_scores = [], []
            device = next(self.parameters()).device
            self.eval()
            with torch.no_grad():
                for x, y in self.test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    _, logits = self.forward(x)
                    # 多分类：保留完整的 softmax 概率向量，形状为 (batch, num_classes)
                    y_score_test = torch.softmax(logits, dim=-1)
                    all_labels.append(y.cpu())
                    all_scores.append(y_score_test.cpu())

            labels_t = torch.cat(all_labels, dim=0).numpy()       # (N,)
            scores_t = torch.cat(all_scores, dim=0).numpy()       # (N, num_classes)
            preds_t = scores_t.argmax(axis=-1)

            # Debug 打印，确认形状和类型，便于排查问题
            print(f"[Debug][Test] labels_t shape: {labels_t.shape}, dtype: {labels_t.dtype}")
            print(f"[Debug][Test] scores_t shape: {scores_t.shape}, dtype: {scores_t.dtype}")
            print(f"[Debug][Test] preds_t type: {type(preds_t)}, shape: {np.shape(preds_t)}")

            test_metrics = {
                "accuracy": float(accuracy_score(labels_t, preds_t)),
                "balanced_accuracy": float(balanced_accuracy_score(labels_t, preds_t)),
                "cohen_kappa": float(cohen_kappa_score(labels_t, preds_t)),
                "f1_macro": float(f1_score(labels_t, preds_t, average="macro")),
                "f1_weighted": float(f1_score(labels_t, preds_t, average="weighted")),
                "f1_micro": float(f1_score(labels_t, preds_t, average="micro")),
            }
            cm_t = confusion_matrix(labels_t, preds_t)

            print(f"[Epoch {epoch}] Test  - Acc: {test_metrics.get('accuracy',0):.4f}, "
                  f"BAcc: {test_metrics.get('balanced_accuracy',0):.4f}")
            print(f"Test Confusion Matrix:\n{cm_t}")

            epoch_test_file = self.output_dir / f"epoch_{epoch}_test.json"
            epoch_test_data = {
                "epoch": epoch,
                "metrics": test_metrics,
                "confusion_matrix": cm_t.tolist(),
            }
            with open(epoch_test_file, "w") as f:
                json.dump(epoch_test_data, f, indent=2)
        
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        # 如果loss_fn是None，创建默认的CrossEntropyLoss
        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        
        # 保存整条概率向量用于 epoch 结束时的评估（多分类）
        y_score = torch.softmax(logit, dim=-1)
        self.running_scores["valid"].append(
            (label.clone().detach().cpu(), y_score.clone().detach().cpu())
        )

        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        return loss
    def on_train_epoch_start(self) -> None:
        self.running_scores["train"]=[]
        return super().on_train_epoch_start()
    def on_train_epoch_end(self) -> None:
        label, y_score = [], []
        for x,y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)        # (N,)
        y_score = torch.cat(y_score, dim=0)    # (N, num_classes)
        
        # 计算训练集多分类指标
        y_pred = torch.argmax(y_score, dim=-1).numpy()
        label_np = label.numpy()
        
        metric_list = [
            "accuracy",
            "balanced_accuracy",
            "cohen_kappa",
            "f1_macro",
            "f1_weighted",
            "f1_micro",
        ]
        results = get_metrics(y_score.numpy(), label_np, metric_list, is_binary=False)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(label_np, y_pred)
        
        # 打印训练指标
        epoch = self.current_epoch
        acc = results.get('accuracy', 0)
        bacc = results.get('balanced_accuracy', 0)
        print(f"[Epoch {epoch}] Train - Acc: {acc:.4f}, BAcc: {bacc:.4f}")
        
        # 保存训练集结果到文件（与valid/test保持一致）
        if self.output_dir is not None:
            epoch_file = self.output_dir / f"epoch_{epoch}_train.json"
            epoch_data = {
                "epoch": epoch,
                "metrics": results,
                "confusion_matrix": cm.tolist(),
            }
            with open(epoch_file, 'w') as f:
                json.dump(epoch_data, f, indent=2)
        
        return super().on_train_epoch_end()
    
    def configure_optimizers(self):
        # 线性 probe：只训练头部；全微调：encoder + 头部一起训练
        if self.freeze_encoder:
            optimizer = torch.optim.AdamW(
                list(self.chan_conv.parameters())+
                list(self.classifier.parameters()),
                weight_decay=weight_decay)

            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=max_epochs,
                pct_start=0.2,
            )
        else:
            encoder_params = list(self.target_encoder.parameters())
            head_params = (
                list(self.chan_conv.parameters())+
                list(self.classifier.parameters())
            )
            encoder_lr = max_lr * self.encoder_lr_ratio
            head_lr = max_lr
            param_groups = [
                {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
                {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
            ]
            optimizer = torch.optim.AdamW(param_groups)
            # OneCycleLR 支持为每个 param_group 设定不同的 max_lr
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[encoder_lr, head_lr],
                steps_per_epoch=steps_per_epoch,
                epochs=max_epochs,
                pct_start=0.2,
            )
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
def _count_classes(paths, num_classes: int = 6, expected_channels: int = 20):
    """对一批文件路径统计类别分布（只统计expected_channels通道的文件）"""
    class_counts = [0] * num_classes
    n_tot = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            X = np.asarray(obj["signal"], dtype=np.float32)
            # 只统计expected_channels通道的文件
            if X.shape[0] == expected_channels:
                y = int(obj["label"])
                if 0 <= y < num_classes:
                    class_counts[y] += 1
                    n_tot += 1
        except Exception:
            continue
    return class_counts, n_tot


def class_stats(folder: str, num_classes: int = 6, expected_channels: int = 20):
    """计算类别分布，用于计算类别权重（只统计expected_channels通道的文件）"""
    paths = [p for ext in ("*.pickle", "*.pkl", "*.pql")
             for p in glob.glob(os.path.join(folder, ext))]
    return _count_classes(paths, num_classes=num_classes, expected_channels=expected_channels)


def check_experiment_complete(batch_size, max_lr, weight_decay, freeze_encoder, 
                               required_epochs=40):
    """
    检查实验是否已经完整训练（0到required_epochs的epoch都有valid结果）
    
    Args:
        batch_size: 批次大小
        max_lr: 学习率
        weight_decay: 权重衰减
        freeze_encoder: 是否冻结encoder
        required_epochs: 需要检查的epoch数量（默认40，即0-40共41个epoch）
    
    Returns:
        (is_complete, output_dir): 是否完整，输出目录路径
    """
    run_name = (
        f"{TASK_NAME}_bs{batch_size}_lr{max_lr}_wd{weight_decay}"
        f"_freeze{int(freeze_encoder)}"
    )
    output_dir = Path("output") / run_name
    
    if not output_dir.exists():
        return False, output_dir
    
    # 检查0到required_epochs的所有epoch的valid文件是否存在
    for epoch in range(required_epochs + 1):
        valid_file = output_dir / f"epoch_{epoch}_valid.json"
        if not valid_file.exists():
            return False, output_dir
    
    return True, output_dir


def main():
    # 重新设定种子（实验级别）
    seed_torch(9)
    # 创建输出文件夹（每次运行一个实验，一个文件夹，不包含时间戳）
    run_name = (
        f"{TASK_NAME}_bs{batch_size}_lr{max_lr}_wd{weight_decay}"
        f"_freeze{int(freeze_encoder)}"
    )
    output_dir = Path("output") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出文件夹: {output_dir.absolute()}")

    # 保存训练配置
    config = {
        "task_name": TASK_NAME,
        "run_name": run_name,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "max_lr": max_lr,
        "freeze_encoder": freeze_encoder,
        "encoder_lr_ratio": encoder_lr_ratio,
        "channels": len(use_channels_names),
        "num_classes": num_classes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 保留在config中，但不在目录名中
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ==================== 数据加载 ====================
    # 计算类别权重（处理类别不平衡）
    train_dir = os.path.join(data_root, "train")
    class_counts, n_tot = class_stats(train_dir, num_classes=num_classes, expected_channels=20)
    # 计算类别权重：逆频率加权
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(n_tot / (num_classes * count))
        else:
            class_weights.append(1.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"[data] train samples={n_tot}, class_counts={class_counts}")
    print(f"[data] class_weights={class_weights.tolist()}")
    
    train_dataset = KaggleEEGDataset(data_root, split="train")
    valid_dataset = KaggleEEGDataset(data_root, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # ==================== 模型初始化 ====================
    model = LitEEGPTCausal(
        freeze_encoder=freeze_encoder,
        encoder_lr_ratio=encoder_lr_ratio,
    )
    model.output_dir = output_dir  # 设置输出目录
    
    # 设置类别权重和损失函数
    device = next(model.parameters()).device
    class_weights_device = class_weights.to(device)
    model.class_weights = class_weights_device
    model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_device)

    # 如果有 test 集，提前构建 test_loader，供每个 epoch 使用
    test_dir = Path(data_root) / "test"
    if test_dir.is_dir():
        print("检测到 test 集，将在每个 epoch 结束后同时评估 test。")
        test_dataset = KaggleEEGDataset(data_root, split="test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
        model.test_loader = test_loader
    else:
        print("未找到 test 目录，将只评估 train/val。")

    # ==================== 训练器配置 ====================
    # 这里不再使用 LearningRateMonitor（它需要 logger），只用最简 callbacks
    callbacks = []

    # 计算 steps_per_epoch（用于 OneCycleLR）
    global steps_per_epoch
    steps_per_epoch = math.ceil(len(train_loader))

    trainer_kwargs = dict(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_checkpointing=False,  # 不保存checkpoint
        logger=False,                # 不使用默认logger，我们自己记录
        enable_progress_bar=True,    # 显示进度条
    )
    if devices is not None:
        trainer_kwargs["devices"] = devices

    trainer = pl.Trainer(**trainer_kwargs)

    # ==================== 开始训练 ====================
    print(f"\n开始训练: {run_name}")
    print(f"模式: {'线性探测' if freeze_encoder else '全局微调'}")
    print(f"Epochs: {max_epochs}, Batch size: {batch_size}, LR: {max_lr}\n")

    trainer.fit(model, train_loader, valid_loader)

    print("\n训练完成！结果保存在:", output_dir.absolute())


class BestEpochTracker(pl.Callback):
    """[LOSO 新增] 纯观察者：记录验证集 valid_balanced_accuracy 最好的那个 epoch
    （1-indexed），不修改 LitEEGPTCausal 的任何训练/验证逻辑，只读
    trainer.callback_metrics。仅在 subject_independent 20折路径下挂载。"""

    def __init__(self, monitor="valid_balanced_accuracy"):
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


def save_eegpt_fold_results(args, lightning_model, checkpoint_callback, test_files,
                             best_epoch, train_time_sec, save_dir):
    """[LOSO 新增] 用被选中的最佳 epoch 权重（ModelCheckpoint 按 valid_balanced_accuracy
    选出的那个 checkpoint，不是训练循环结束时的最后一轮权重）对 test 集重新推理一次，保存：
      {task}_{model}_fold{i:02d}.npz  -- sample_id / y_true / y_pred / y_prob / subject_id
      {task}_{model}_fold{i:02d}.json -- fold / test_subject / val_subject / balanced_accuracy /
                                          best_epoch / hyperparams / train_time_sec /
                                          peak_gpu_mem_mb / gpu_name
    失败直接抛异常，不静默跳过；已存在的旧结果先改名备份，不会被静默覆盖。"""
    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        raise RuntimeError(
            f"save_eegpt_fold_results: no best checkpoint found (best_model_path={best_ckpt_path!r})"
        )
    state = torch.load(best_ckpt_path, map_location="cpu")
    lightning_model.load_state_dict(state["state_dict"])

    device = next(lightning_model.parameters()).device
    lightning_model.eval()

    sid_test_dataset = KaggleEEGDataset(data_root, split="test", file_list=test_files, return_sample_id=True)
    sid_test_loader = torch.utils.data.DataLoader(
        sid_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_motion_with_sample_id,
    )

    subj_re = re.compile(r'^S(\d+)_')
    sample_ids, y_true, y_pred, y_prob, subject_ids = [], [], [], [], []
    with torch.no_grad():
        for X, Y, batch_sample_ids in sid_test_loader:
            X = X.to(device)
            _, logits = lightning_model(X)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
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
        raise RuntimeError("save_eegpt_fold_results: no samples collected, refusing to save an empty file.")

    sample_ids_arr = np.array(sample_ids)
    order = np.argsort(sample_ids_arr)
    sample_ids_arr = sample_ids_arr[order]
    y_true_arr = np.array(y_true, dtype=np.int64)[order]
    y_pred_arr = np.array(y_pred, dtype=np.int64)[order]
    y_prob_arr = np.array(y_prob, dtype=np.float32)[order]
    subject_id_arr = np.array(subject_ids, dtype=np.int64)[order]

    task = args.task_name or "motion"
    model_name = args.model_name or "eegpt"
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{task}_{model_name}_fold{args.fold_idx:02d}.npz")
    json_path = os.path.join(save_dir, f"{task}_{model_name}_fold{args.fold_idx:02d}.json")

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
        raise RuntimeError(f"save_eegpt_fold_results: failed to write {npz_path}")
    _reload = np.load(npz_path)
    for key in ("sample_id", "y_true", "y_pred", "y_prob", "subject_id"):
        if key not in _reload:
            raise RuntimeError(f"save_eegpt_fold_results: {npz_path} missing key {key!r} after write")
        if len(_reload[key]) != len(sample_ids_arr):
            raise RuntimeError(f"save_eegpt_fold_results: {npz_path} key {key!r} length mismatch after write")

    balanced_accuracy = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    hyperparams = {
        "lr": max_lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": max_epochs,
        "optimizer": "AdamW",
        "seed": 9,
        "freeze_encoder": bool(freeze_encoder),
        "encoder_lr_ratio": encoder_lr_ratio,
        "channels": len(use_channels_names),
        "num_classes": num_classes,
    }

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated(0) / (1024 ** 2)) if torch.cuda.is_available() else None

    meta = {
        "fold": args.fold_idx,
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
        raise RuntimeError(f"save_eegpt_fold_results: failed to write {json_path}")
    with open(json_path) as f:
        json.load(f)  # 回读校验 JSON 没写坏

    print(f"Saved fold predictions to {npz_path}")
    print(f"Saved fold metadata to {json_path}")
    print(f"  fold={args.fold_idx} test={args.test_subject} val={args.val_subject} "
          f"best_epoch={best_epoch} balanced_accuracy={balanced_accuracy:.5f}")


def run_loso_fold(args):
    """[LOSO 新增] 20折 subject-independent LOSO 单折训练入口：
    test = args.test_subject, val = args.val_subject, train = 其余受试者。
    固定超参数（不做HPO）：lr/weight_decay/batch_size/epochs 见 LOSO_* 常量，
    freeze_encoder=False（全局微调，非 linear probe）。
    不改动 LitEEGPTCausal 的模型定义/训练逻辑/优化器逻辑，只是复用它们并额外
    挂载按 valid_balanced_accuracy 选模的 checkpoint + 事后保存 npz/json。"""
    global batch_size, max_lr, weight_decay, max_epochs, freeze_encoder, encoder_lr_ratio, steps_per_epoch

    if not args.test_subject or not args.val_subject:
        raise ValueError("split_mode=subject_independent requires --test_subject and --val_subject")
    if args.test_subject == args.val_subject:
        raise ValueError("--test_subject and --val_subject must be different")

    seed_torch(9)

    subject_to_files = list_motion_files_by_subject(data_root)
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

    print("=" * 70)
    print(f"[split_mode=subject_independent] fold={args.fold_idx}  "
          f"test={args.test_subject}  val={args.val_subject}  train={len(train_subjects)} subjects")
    print(f"  All subjects ({len(subjects)}): {subjects}")
    print(f"  file counts: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print("=" * 70)

    # 固定超参数：不做HPO，全局微调（full finetune）
    batch_size = LOSO_BATCH_SIZE
    max_lr = LOSO_LR
    weight_decay = LOSO_WEIGHT_DECAY
    max_epochs = LOSO_EPOCHS
    freeze_encoder = LOSO_FREEZE_ENCODER
    encoder_lr_ratio = LOSO_ENCODER_LR_RATIO

    run_tag = f"fold{args.fold_idx:02d}_test{args.test_subject}"
    output_dir = Path("output") / "EEGPT_Motor_LOSO" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出文件夹: {output_dir.absolute()}")

    config = {
        "task_name": TASK_NAME,
        "run_name": run_tag,
        "split_mode": "subject_independent",
        "fold_idx": args.fold_idx,
        "test_subject": args.test_subject,
        "val_subject": args.val_subject,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "max_lr": max_lr,
        "weight_decay": weight_decay,
        "freeze_encoder": freeze_encoder,
        "encoder_lr_ratio": encoder_lr_ratio,
        "channels": len(use_channels_names),
        "num_classes": num_classes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    class_counts, n_tot = _count_classes(train_files, num_classes=num_classes, expected_channels=20)
    class_weights = torch.tensor(
        [n_tot / (num_classes * c) if c > 0 else 1.0 for c in class_counts],
        dtype=torch.float32,
    )
    print(f"[data] fold={args.fold_idx} train samples={n_tot}, class_counts={class_counts}")
    print(f"[data] class_weights={class_weights.tolist()}")

    train_dataset = KaggleEEGDataset(data_root, split="train", file_list=train_files)
    valid_dataset = KaggleEEGDataset(data_root, split="val", file_list=val_files)
    test_dataset = KaggleEEGDataset(data_root, split="test", file_list=test_files)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True,
    )

    model = LitEEGPTCausal(
        freeze_encoder=freeze_encoder,
        encoder_lr_ratio=encoder_lr_ratio,
    )
    model.output_dir = output_dir

    device = next(model.parameters()).device
    class_weights_device = class_weights.to(device)
    model.class_weights = class_weights_device
    model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_device)
    model.test_loader = test_loader

    steps_per_epoch = math.ceil(len(train_loader))

    # [LOSO 新增] 仅这条路径打开 checkpointing，按 valid_balanced_accuracy 选模；
    # 旧的网格搜索路径（main()）保持 enable_checkpointing=False 不变。
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        monitor="valid_balanced_accuracy",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{valid_balanced_accuracy:.5f}",
        auto_insert_metric_name=False,
    )
    best_epoch_tracker = BestEpochTracker(monitor="valid_balanced_accuracy")

    trainer_kwargs = dict(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, best_epoch_tracker],
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=True,
    )
    if devices is not None:
        trainer_kwargs["devices"] = devices

    trainer = pl.Trainer(**trainer_kwargs)

    print(f"\n开始训练: {run_tag}")
    print(f"模式: {'线性探测' if freeze_encoder else '全局微调'}")
    print(f"Epochs: {max_epochs}, Batch size: {batch_size}, LR: {max_lr}, WD: {weight_decay}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start_time = time.time()
    trainer.fit(model, train_loader, valid_loader)
    train_time_sec = time.time() - train_start_time

    print(f"\n训练完成，最佳 epoch={best_epoch_tracker.best_epoch}（valid_balanced_accuracy={best_epoch_tracker.best_score:.5f}）")

    fold_results_dir = args.fold_results_dir or "./fold_results_eegpt"
    save_eegpt_fold_results(
        args, model, checkpoint_callback, test_files,
        best_epoch_tracker.best_epoch, train_time_sec, save_dir=fold_results_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_mode", type=str, default="random_epoch",
        choices=["random_epoch", "subject_independent"],
        help="random_epoch(默认) = 旧的网格搜索流程，读取 Motiondata/{train,val,test} 固定划分；"
             "subject_independent = 20折 LOSO 单折训练，需要 --test_subject/--val_subject。"
             "不传本参数时行为与之前完全一致。"
    )
    parser.add_argument("--test_subject", type=str, default=None,
                        help="e.g. Sub04；split_mode=subject_independent 时必填")
    parser.add_argument("--val_subject", type=str, default=None,
                        help="e.g. Sub05；split_mode=subject_independent 时必填")
    parser.add_argument("--fold_idx", type=int, default=0,
                        help="LOSO fold 序号（0-based），用于输出目录名/保存文件名")
    parser.add_argument("--model_name", type=str, default="eegpt",
                        help="保存 {task}_{model}_fold{i}.npz/json 时使用的模型名")
    parser.add_argument("--task_name", type=str, default="motion",
                        help="保存 {task}_{model}_fold{i}.npz/json 时使用的任务名")
    parser.add_argument("--fold_results_dir", type=str, default=None,
                        help="npz/json 保存目录；默认 ./fold_results_eegpt")
    cli_args = parser.parse_args()

    if cli_args.split_mode == "subject_independent":
        run_loso_fold(cli_args)
    else:
        # ===== [保留原逻辑] 网格搜索：遍历所有 (batch_size, lr, weight_decay) 组合 =====
        for bs in BS_LIST:
            for lr in LR_LIST:
                for wd in WD_LIST:
                    print("\n" + "=" * 80)
                    print(f"检查实验: bs={bs}, lr={lr}, wd={wd}, freeze_encoder={freeze_encoder}")
                    print("=" * 80)

                    # 检查实验是否已经完整训练（0-40 epoch）
                    is_complete, output_dir = check_experiment_complete(
                        bs, lr, wd, freeze_encoder, required_epochs=40
                    )

                    if is_complete:
                        print(f"✓ 实验已完成（0-40 epoch），跳过: {output_dir.name}")
                        print("=" * 80)
                        continue

                    # 检查是否有部分结果
                    if output_dir.exists():
                        existing_epochs = []
                        for epoch_file in sorted(output_dir.glob("epoch_*_valid.json")):
                            try:
                                epoch_num = int(epoch_file.stem.split("_")[1])
                                existing_epochs.append(epoch_num)
                            except:
                                pass
                        if existing_epochs:
                            max_epoch = max(existing_epochs)
                            print(f"⚠ 检测到已有部分结果（最高到 epoch {max_epoch}），将重新训练并覆盖")
                        else:
                            print(f"→ 实验目录存在但无结果文件，开始训练...")
                    else:
                        print(f"→ 实验不存在，开始新训练...")
                    print("=" * 80)

                    # 更新全局超参数
                    batch_size = bs
                    max_lr = lr
                    weight_decay = wd

                    # 运行一次完整实验（会自动创建带超参信息的输出文件夹，如果已存在会覆盖）
                    main()

