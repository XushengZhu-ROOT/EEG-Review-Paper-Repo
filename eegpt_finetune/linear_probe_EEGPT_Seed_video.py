import random 
import os
import math
import torch
from torch import nn
import pytorch_lightning as pl
from pathlib import Path
import json
from datetime import datetime

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
import re


# ==================== 数据相关配置 ====================
# SEED Emotion 任务：
# - 六分类（0,1,2,3,4,5），排除neutral(2)，重新映射：happy(0), sad(1), disgust(2), fear(3), surprise(4), anger(5)
# - 4 秒，采样率 256 Hz -> 时间长度 1024
# - 原始 62 通道，移除 CB1 和 CB2（索引57和61），得到 60 通道

use_channels_names = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'O1', 'OZ', 'O2'
]  # 60个通道（排除CB1和CB2）

# ==================== 用户可配置参数 ====================
# 建议你只改这一段就能完成大部分自定义

# 任务名称（会用于输出文件夹命名）
TASK_NAME = "EEGPT_Seed_video"

# 数据路径（SEED 数据目录位置，建议结构为 seed_data/{train,val,test}，数据在子文件夹内）
data_root = "./seed_data"

# 分类类别数（SEED Emotion: 0~5 六分类，排除neutral）
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

# ====== 超参数搜索范围（可选）======
BS_LIST = [128, 64, 32]
LR_LIST = [1e-3, 4e-4, 1e-4]
WD_LIST = [1e-2, 1e-3]


# 标签映射函数：排除neutral(2)，重新映射到0-5
def remap_label(original_label: int) -> int:
    """Remap label excluding neutral (2)."""
    if original_label == 2:  # neutral - should be filtered out
        return None
    elif original_label < 2:
        return original_label  # happy(0), sad(1) -> 0, 1
    else:
        return original_label - 1  # disgust(3), fear(4), surprise(5), anger(6) -> 2, 3, 4, 5

# 视频级别评估辅助函数
def extract_video_index(epoch_id: str) -> int:
    """从 epoch_id 中提取 video_index"""
    if epoch_id is None:
        return None
    match = re.search(r'video_index_(\d+)_chunk', epoch_id)
    if match:
        return int(match.group(1))
    return None

def extract_subject_id(epoch_id: str) -> int:
    """从 epoch_id 中提取 subject_id"""
    if epoch_id is None:
        return None
    match = re.search(r'subject_(\d+)_', epoch_id)
    if match:
        return int(match.group(1))
    return None

def majority_vote_with_tie_handling(predictions: np.ndarray, true_label: int):
    """
    多数投票，处理平票情况
    
    Args:
        predictions: 同一视频所有chunks的预测结果 (n_chunks,)
        true_label: 真实标签
    
    Returns:
        (predicted_label, score)
        score: 1.0 (完全正确), 0.5 (平票且真实标签在候选中), 0.0 (错误)
    """
    if len(predictions) == 0:
        return -1, 0.0
    
    # 统计每个类别的投票数
    unique, counts = np.unique(predictions, return_counts=True)
    max_count = np.max(counts)
    
    # 找出得票最多的类别（可能有多个）
    winners = unique[counts == max_count]
    
    if len(winners) == 1:
        # 单一获胜者
        predicted_label = winners[0]
        score = 1.0 if predicted_label == true_label else 0.0
    else:
        # 平票情况
        if true_label in winners:
            # 真实标签在平票候选中，选择真实标签，得0.5分
            predicted_label = true_label
            score = 0.5
        else:
            # 真实标签不在平票候选中，选择第一个候选项，得0.0分
            predicted_label = winners[0]
            score = 0.0
    
    return predicted_label, score


class SEEDEEGDataset(torch.utils.data.Dataset):
    """
    从 seed_data/{split}/**/*.pickle 递归读取数据（支持子文件夹）。
    每个 pickle 是一个 dict:
        - 'signal': ndarray, shape (62, 1024)
        - 'label' : int, 0-6 (排除neutral=2)
        - 'epoch_id': str (用于视频级别评估)
    """
    def __init__(self, root_dir: str, split: str = "train", expected_channels: int = 62, target_channels: int = 60):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        assert os.path.isdir(self.split_dir), f"{self.split_dir} 不存在"
        
        self.expected_channels = expected_channels
        self.target_channels = target_channels
        # CB1和CB2在62通道数据中的索引（0-based）
        self.cb_indices = [57, 61]  # CB1 at 57, CB2 at 61

        # 递归查找所有pickle文件
        all_paths = sorted([
            p for ext in ("*.pickle", "*.pkl", "*.pql")
            for p in glob.glob(os.path.join(self.split_dir, "**", ext), recursive=True)
        ])
        
        print(f"[SEEDEEGDataset] 在 {self.split_dir} 中找到了 {len(all_paths)} 个pickle文件")
        if len(all_paths) == 0:
            # 尝试直接列出目录内容，帮助调试
            if os.path.isdir(self.split_dir):
                print(f"[SEEDEEGDataset] 目录存在，尝试列出内容...")
                try:
                    items = os.listdir(self.split_dir)
                    print(f"[SEEDEEGDataset] 目录下的直接内容: {items[:10]}...")  # 只显示前10个
                    # 检查是否有子文件夹
                    subdirs = [d for d in items if os.path.isdir(os.path.join(self.split_dir, d))]
                    if subdirs:
                        print(f"[SEEDEEGDataset] 发现子文件夹: {subdirs[:5]}...")
                        # 检查第一个子文件夹中的文件
                        if len(subdirs) > 0:
                            first_subdir = os.path.join(self.split_dir, subdirs[0])
                            subdir_files = [f for f in os.listdir(first_subdir) 
                                          if f.endswith(('.pickle', '.pkl', '.pql'))]
                            print(f"[SEEDEEGDataset] 子文件夹 {subdirs[0]} 中有 {len(subdir_files)} 个pickle文件")
                except Exception as e:
                    print(f"[SEEDEEGDataset] 无法列出目录内容: {e}")
            else:
                print(f"[SEEDEEGDataset] 警告: {self.split_dir} 不是目录或不存在")
        
        # 过滤：只保留expected_channels通道的文件，并排除neutral(2)
        self.paths = []
        skipped_channels = 0
        skipped_neutral = 0
        error_count = 0
        for p in all_paths:
            try:
                with open(p, "rb") as f_obj:
                    obj = pickle.load(f_obj)
                X = np.asarray(obj["signal"], dtype=np.float32)
                y_original = int(obj["label"])
                # 跳过通道数不匹配的文件
                if X.shape[0] != expected_channels:
                    skipped_channels += 1
                    continue
                # 跳过neutral(2)
                if y_original == 2:
                    skipped_neutral += 1
                    continue
                self.paths.append(p)
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # 只打印前3个错误，避免输出太多
                    print(f"[SEEDEEGDataset] 处理文件 {p} 时出错: {e}")
                continue
        
        if len(self.paths) == 0:
            print(f"[SEEDEEGDataset] 错误详情: 总文件={len(all_paths)}, "
                  f"跳过通道不匹配={skipped_channels}, 跳过neutral={skipped_neutral}, 错误={error_count}")
            raise RuntimeError(f"{self.split_dir} 下没有找到任何有效的 {expected_channels} 通道文件（排除neutral）")
        
        if skipped_channels > 0:
            print(f"[SEEDEEGDataset] split={split}, 跳过 {skipped_channels} 个通道数不匹配的文件")
        if skipped_neutral > 0:
            print(f"[SEEDEEGDataset] split={split}, 跳过 {skipped_neutral} 个neutral(2)样本")
        print(f"[SEEDEEGDataset] split={split}, 样本数={len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with open(path, "rb") as f:
            obj = pickle.load(f)

        x = obj["signal"]   # (62, 1024) 原始 62 通道
        y_original = int(obj["label"])  # 原始标签 0-6 (排除2)
        epoch_id = obj.get("epoch_id", None)  # 提取epoch_id用于视频级别评估

        # 检查通道数
        if x.shape[0] != self.expected_channels:
            raise ValueError(f"Expected {self.expected_channels} channels, got {x.shape[0]}")
        
        # 重新映射标签：排除neutral(2)，映射到0-5
        y = remap_label(y_original)
        if y is None:
            raise ValueError(f"Unexpected neutral label (2) in dataset - should have been filtered")
        
        # 移除CB1和CB2通道（索引57和61）
        keep_indices = [i for i in range(x.shape[0]) if i not in self.cb_indices]
        x = x[keep_indices, :]  # (60, 1024)
        
        # 数据清理：NaN/Inf处理
        x = np.nan_to_num(x, posinf=0.0, neginf=0.0)
        
        x = torch.tensor(x, dtype=torch.float32)  # (C, T)
        y = torch.tensor(y, dtype=torch.long)

        return x, y, epoch_id


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
            img_size=[self.chans_num, 4 * 256],  # 60 通道, 4s@256Hz = 1024
            patch_size=32*2,  # 64
            patch_stride=None,  # 不使用patch_stride，按motor的方式
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

        # 输入为 60 通道（移除CB1和CB2后）
        self.chan_conv       = Conv1dWithConstraint(60, self.chans_num, 1, max_norm=1)
        
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
            # 默认计算：假设patch_stride=None，seq_len=1024, patch_size=64
            seq_len = 4 * 256  # 1024
            patch_size = 64
            self.N = seq_len // patch_size  # 1024 // 64 = 16
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
        
    def _evaluate_video_level(self, labels_all, preds_all, epoch_ids_all, split_name="valid"):
        """视频级别的评估（majority voting）"""
        # 按(subject_id, video_index)分组
        video_groups = {}  # (subject_id, video_index) -> [(pred, true_label), ...]
        
        for pred, true_label, epoch_id in zip(preds_all, labels_all, epoch_ids_all):
            if epoch_id is None:
                continue
            subject_id = extract_subject_id(epoch_id)
            video_index = extract_video_index(epoch_id)
            
            if subject_id is None or video_index is None:
                continue
            
            key = (subject_id, video_index)
            if key not in video_groups:
                video_groups[key] = []
            video_groups[key].append((int(pred), int(true_label)))
        
        # 对每个视频进行majority voting
        video_results = []
        for (subject_id, video_index), chunks in video_groups.items():
            predictions = np.array([p for p, _ in chunks])
            true_labels = np.array([t for _, t in chunks])
            
            # 同一视频的所有chunks应该有相同的真实标签
            true_label = int(true_labels[0])
            if not np.all(true_labels == true_label):
                print(f"[warn] Video (subject={subject_id}, video={video_index}) has inconsistent true labels")
            
            # Majority vote
            pred_label, score = majority_vote_with_tie_handling(predictions, true_label)
            video_results.append((subject_id, video_index, pred_label, true_label, score))
        
        if len(video_results) == 0:
            return {"accuracy": 0.0, "balanced_accuracy": 0.0}
        
        # 计算视频级别指标
        video_scores = np.array([score for _, _, _, _, score in video_results])
        video_acc = float(video_scores.mean())
        
        # 按subject统计
        subject_results = {}
        for subject_id, _, _, _, score in video_results:
            if subject_id not in subject_results:
                subject_results[subject_id] = []
            subject_results[subject_id].append(score)
        
        subject_accuracies = {}
        for subject_id, scores in subject_results.items():
            subject_accuracies[subject_id] = float(np.mean(scores))
        
        # 视频级别的类别指标
        video_true = np.array([true_label for _, _, _, true_label, _ in video_results])
        video_pred = np.array([pred_label for _, _, pred_label, _, _ in video_results])
        
        per_class_acc = []
        for c in range(self.num_class):
            mask = (video_true == c)
            if mask.sum() > 0:
                per_class_acc.append(float((video_pred[mask] == c).sum() / mask.sum()))
            else:
                per_class_acc.append(0.0)
        
        bacc = float(np.mean(per_class_acc))
        
        cm_video = confusion_matrix(video_true, video_pred, labels=list(range(self.num_class)))
        
        return {
            "accuracy": video_acc,
            "balanced_accuracy": bacc,
            "per_class_acc": per_class_acc,
            "confusion_matrix": cm_video.tolist(),
            "subject_accuracies": subject_accuracies,
            "n_videos": len(video_results),
            "n_subjects": len(subject_accuracies),
        }
    
    def forward(self, x):
        B, C, T = x.shape
        # SEED数据预处理：按motor的方式，不进行temporal_interpolation（数据已经是1024）
        x = x / 100.0  # 归一化
        # x = x - x.mean(dim=-2, keepdim=True)  # 去均值
        # 不需要temporal_interpolation，数据已经是4s@256Hz=1024
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
        if len(batch) == 3:
            x, y, epoch_ids = batch  # SEED数据包含epoch_id
        else:
            x, y = batch
            epoch_ids = None
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
            
        label, y_score, epoch_ids_all = [], [], []
        for item in self.running_scores["valid"]:
            if len(item) == 3:
                x, y, epoch_ids_batch = item
                epoch_ids_all.extend(epoch_ids_batch)
            else:
                x, y = item
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
        
        print(f"\n[Epoch {epoch}] Validation (Chunk-level) - Acc: {acc:.4f}, BAcc: {bacc:.4f}")
        print(f"Chunk-level Confusion Matrix:\n{cm}")

        # Video-level 评估（如果有epoch_id信息）
        video_valid_metrics = None
        if len(epoch_ids_all) > 0 and any(eid is not None for eid in epoch_ids_all):
            video_valid_metrics = self._evaluate_video_level(label_np, y_pred, epoch_ids_all, "valid")
            print(f"[Epoch {epoch}] Validation (Video-level) - Acc: {video_valid_metrics.get('accuracy',0):.4f}, "
                  f"BAcc: {video_valid_metrics.get('balanced_accuracy',0):.4f}")
            if video_valid_metrics.get("confusion_matrix") is not None:
                cm_video = np.array(video_valid_metrics.get("confusion_matrix"))
                print(f"Video-level Confusion Matrix:\n{cm_video}")
            if video_valid_metrics.get("subject_accuracies"):
                subject_str = ", ".join([f"s{subj}:{acc:.3f}" for subj, acc in sorted(video_valid_metrics["subject_accuracies"].items())])
                print(f"  [Subject accuracies] {subject_str}")
        
        # 保存到文件
        if self.output_dir is not None:
            epoch_file = self.output_dir / f"epoch_{epoch}_valid.json"
            epoch_data = {
                "epoch": epoch,
                "chunk_level": {
                    "metrics": results,
                    "confusion_matrix": cm.tolist(),
                },
            }
            # 如果有视频级别评估结果，也保存
            if video_valid_metrics is not None:
                epoch_data["video_level"] = {
                    "metrics": video_valid_metrics,
                    "confusion_matrix": video_valid_metrics.get("confusion_matrix", []),
                    "subject_accuracies": video_valid_metrics.get("subject_accuracies", {}),
                }
            with open(epoch_file, 'w') as f:
                json.dump(epoch_data, f, indent=2)

        # ---------- 每个 epoch 额外在 test 集上评估（如果提供了 test_loader） ----------
        if self.test_loader is not None and self.output_dir is not None:
            all_labels, all_scores, all_epoch_ids = [], [], []
            device = next(self.parameters()).device
            self.eval()
            with torch.no_grad():
                for batch in self.test_loader:
                    if len(batch) == 3:
                        x, y, epoch_ids = batch
                    else:
                        x, y = batch
                        epoch_ids = [None] * len(y)
                    x = x.to(device)
                    y = y.to(device)
                    _, logits = self.forward(x)
                    # 多分类：保留完整的 softmax 概率向量，形状为 (batch, num_classes)
                    y_score_test = torch.softmax(logits, dim=-1)
                    all_labels.append(y.cpu())
                    all_scores.append(y_score_test.cpu())
                    all_epoch_ids.extend(epoch_ids)

            labels_t = torch.cat(all_labels, dim=0).numpy()       # (N,)
            scores_t = torch.cat(all_scores, dim=0).numpy()       # (N, num_classes)
            preds_t = scores_t.argmax(axis=-1)

            # Chunk-level 评估
            test_metrics = {
                "accuracy": float(accuracy_score(labels_t, preds_t)),
                "balanced_accuracy": float(balanced_accuracy_score(labels_t, preds_t)),
                "cohen_kappa": float(cohen_kappa_score(labels_t, preds_t)),
                "f1_macro": float(f1_score(labels_t, preds_t, average="macro")),
                "f1_weighted": float(f1_score(labels_t, preds_t, average="weighted")),
                "f1_micro": float(f1_score(labels_t, preds_t, average="micro")),
            }
            cm_t = confusion_matrix(labels_t, preds_t)

            print(f"[Epoch {epoch}] Test (Chunk-level) - Acc: {test_metrics.get('accuracy',0):.4f}, "
                  f"BAcc: {test_metrics.get('balanced_accuracy',0):.4f}")
            print(f"Test Chunk-level Confusion Matrix:\n{cm_t}")

            # Video-level 评估（majority voting）
            video_test_metrics = self._evaluate_video_level(labels_t, preds_t, all_epoch_ids, "test")
            print(f"[Epoch {epoch}] Test (Video-level) - Acc: {video_test_metrics.get('accuracy',0):.4f}, "
                  f"BAcc: {video_test_metrics.get('balanced_accuracy',0):.4f}")
            if video_test_metrics.get("confusion_matrix") is not None:
                cm_video_test = np.array(video_test_metrics.get("confusion_matrix"))
                print(f"Test Video-level Confusion Matrix:\n{cm_video_test}")

            epoch_test_file = self.output_dir / f"epoch_{epoch}_test.json"
            epoch_test_data = {
                "epoch": epoch,
                "chunk_level": {
                    "metrics": test_metrics,
                    "confusion_matrix": cm_t.tolist(),
                },
                "video_level": {
                    "metrics": video_test_metrics,
                    "confusion_matrix": video_test_metrics.get("confusion_matrix", []),
                    "subject_accuracies": video_test_metrics.get("subject_accuracies", {}),
                },
            }
            with open(epoch_test_file, "w") as f:
                json.dump(epoch_test_data, f, indent=2)
        
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, epoch_ids = batch  # SEED数据包含epoch_id
        else:
            x, y = batch
            epoch_ids = None
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
        # 保存epoch_id信息用于视频级别评估
        if len(batch) == 3:
            _, _, epoch_ids_batch = batch
        else:
            epoch_ids_batch = [None] * len(label)
        self.running_scores["valid"].append(
            (label.clone().detach().cpu(), y_score.clone().detach().cpu(), epoch_ids_batch)
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
        
def class_stats(folder: str, num_classes: int = 6, expected_channels: int = 62):
    """计算类别分布，用于计算类别权重（只统计expected_channels通道的文件，排除neutral=2）"""
    # 递归查找所有pickle文件
    paths = [p for ext in ("*.pickle", "*.pkl", "*.pql")
             for p in glob.glob(os.path.join(folder, "**", ext), recursive=True)]
    
    print(f"[class_stats] 在 {folder} 中找到了 {len(paths)} 个pickle文件")
    if len(paths) == 0:
        # 尝试直接列出目录内容，帮助调试
        if os.path.isdir(folder):
            print(f"[class_stats] 目录存在，尝试列出内容...")
            try:
                items = os.listdir(folder)
                print(f"[class_stats] 目录下的直接内容: {items[:10]}...")  # 只显示前10个
            except Exception as e:
                print(f"[class_stats] 无法列出目录内容: {e}")
        else:
            print(f"[class_stats] 警告: {folder} 不是目录或不存在")
    
    class_counts = [0] * num_classes
    n_tot = 0
    skipped_channels = 0
    skipped_neutral = 0
    error_count = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            X = np.asarray(obj["signal"], dtype=np.float32)
            # 只统计expected_channels通道的文件
            if X.shape[0] != expected_channels:
                skipped_channels += 1
                continue
            y_original = int(obj["label"])
            # 跳过neutral(2)
            if y_original == 2:
                skipped_neutral += 1
                continue
            # 重新映射标签
            y = remap_label(y_original)
            if y is not None and 0 <= y < num_classes:
                class_counts[y] += 1
                n_tot += 1
        except Exception as e:
            error_count += 1
            if error_count <= 3:  # 只打印前3个错误，避免输出太多
                print(f"[class_stats] 处理文件 {p} 时出错: {e}")
            continue
    
    if len(paths) > 0:
        print(f"[class_stats] 处理结果: 总文件={len(paths)}, 有效样本={n_tot}, "
              f"跳过通道不匹配={skipped_channels}, 跳过neutral={skipped_neutral}, 错误={error_count}")
    
    return class_counts, n_tot


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
    class_counts, n_tot = class_stats(train_dir, num_classes=num_classes, expected_channels=62)
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
    
    train_dataset = SEEDEEGDataset(data_root, split="train", expected_channels=62, target_channels=60)
    valid_dataset = SEEDEEGDataset(data_root, split="val", expected_channels=62, target_channels=60)

    # 自定义collate函数以支持epoch_id
    def collate_fn(batch):
        if len(batch[0]) == 3:
            xs, ys, epoch_ids = zip(*batch)
            return torch.stack(xs, 0), torch.stack(ys, 0), list(epoch_ids)
        else:
            xs, ys = zip(*batch)
            return torch.stack(xs, 0), torch.stack(ys, 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
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
        test_dataset = SEEDEEGDataset(data_root, split="test", expected_channels=62, target_channels=60)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
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


if __name__ == "__main__":
    # 网格搜索：遍历所有 (batch_size, lr, weight_decay) 组合
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

