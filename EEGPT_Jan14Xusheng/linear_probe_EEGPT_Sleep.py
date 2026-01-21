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
# Sleep 任务：
# - 五分类（0,1,2,3,4）
# - 30 秒，采样率 256 Hz -> 时间长度 7680 (30*256)
# - 6 通道：['C3', 'C4', 'F3', 'F4', 'O1', 'O2']

use_channels_names = [
    'C3', 'C4', 'F3', 'F4', 'O1', 'O2'  # 6个通道
]

# ==================== 用户可配置参数 ====================
# 建议你只改这一段就能完成大部分自定义

# 任务名称（会用于输出文件夹命名）
TASK_NAME = "EEGPT_Sleep"

# 数据路径（Sleep 数据目录位置，建议结构为 sleep_data/{train,val,test}）
data_root = "sleep_data"

# 分类类别数（Sleep: 0~4 五分类）
num_classes = 5

# 训练超参数（默认值，单次跑 main() 会用到）
batch_size = 64
max_epochs = 50
max_lr = 4e-4
weight_decay = 1e-2  # 权重衰减（用于 AdamW）

# 模型微调方式（默认值）
freeze_encoder = False      # True = 线性探测（只训头部），False = 全局微调
encoder_lr_ratio = 0.1      # 全微调时 encoder 学习率 = max_lr * encoder_lr_ratio

# DataLoader 线程数
num_workers =  4

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


class KaggleEEGDataset(torch.utils.data.Dataset):
    """
    从 sleep_data/{split}/*.pickle 读取数据。
    每个 pickle 是一个 dict:
        - 'signal': ndarray, shape (6, 7680)  # 6通道，30秒@256Hz
        - 'label' : int, 0~4  # 5分类
        - 'epoch_id': str
    """
    def __init__(self, root_dir: str, split: str = "train", expected_channels: int = 6, target_channels: int = 6):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        assert os.path.isdir(self.split_dir), f"{self.split_dir} 不存在"
        
        self.expected_channels = expected_channels
        self.target_channels = target_channels

        # 过滤：只保留expected_channels通道的文件
        all_files = [
            os.path.join(self.split_dir, f)
            for f in os.listdir(self.split_dir)
            if f.endswith(".pickle")
        ]
        
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
            raise RuntimeError(f"{self.split_dir} 下没有找到任何 {expected_channels} 通道的 .pickle 文件")
        
        if skipped > 0:
            print(f"[KaggleEEGDataset] split={split}, 跳过 {skipped} 个通道数不匹配的文件")
        print(f"[KaggleEEGDataset] split={split}, 样本数={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, "rb") as f:
            obj = pickle.load(f)

        x = obj["signal"]   # (6, 7680) 原始 6 通道，30秒@256Hz
        y = obj["label"]    # int 0~4

        # 检查通道数
        if x.shape[0] != self.expected_channels:
            raise ValueError(f"Expected {self.expected_channels} channels, got {x.shape[0]}")
        
        # Sleep数据已经就是6通道，不需要丢弃
        x = x[:self.target_channels, :]  # (6, 7680)
        
        # 数据清理：NaN/Inf处理
        x = np.nan_to_num(x, posinf=0.0, neginf=0.0)
        
        x = torch.tensor(x, dtype=torch.float32)  # (C, T)
        y = torch.tensor(y, dtype=torch.long)

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
        self.chans_num = len(use_channels_names)  # 6
        self.num_class = num_classes
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_ratio = encoder_lr_ratio  # encoder 学习率相对于 head 的比例
        # test dataloader（在 main 里赋值，用于每个 epoch 做 test）
        self.test_loader = None

        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 30 * 256],  # 6 通道, 30s@256Hz = 7680
            patch_size=32*2,  # 64
            embed_num=4,
            embed_dim=512,
            depth=7,  # 从8改为7，因为显存不够
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
        
        # 如果模型depth小于预训练模型，需要过滤掉多余的层
        # 例如：depth=7时，需要过滤掉blocks.7的权重（因为blocks索引是0-6，共7层）
        model_depth = len(target_encoder.blocks) if hasattr(target_encoder, 'blocks') else 7
        filtered_stat = {}
        skipped_keys = []
        for k, v in target_encoder_stat.items():
            # 检查是否超出当前模型depth的权重
            # 如果当前模型depth=7（blocks.0到blocks.6），需要跳过blocks.7及以上的权重
            # 解析block索引：blocks.7.xxx -> 7
            if 'blocks.' in k:
                try:
                    # 提取block编号，例如 "blocks.7.norm1.weight" -> 7
                    block_idx = int(k.split('blocks.')[1].split('.')[0])
                    if block_idx >= model_depth:
                        skipped_keys.append(k)
                        continue
                except (ValueError, IndexError):
                    # 如果解析失败，保留该键（可能是其他相关的键）
                    pass
            filtered_stat[k] = v
        
        if skipped_keys and self.debug:
            print(f"[Debug] 跳过 {len(skipped_keys)} 个不匹配的权重键（depth不匹配）")
            if len(skipped_keys) <= 10:
                print(f"[Debug] 跳过的键: {skipped_keys[:10]}")
        
        # 使用strict=False允许部分权重不匹配（用于depth不匹配的情况）
        missing_keys, unexpected_keys = self.target_encoder.load_state_dict(filtered_stat, strict=False)
        if self.debug:
            if missing_keys:
                print(f"[Debug] 缺失的权重键数量: {len(missing_keys)}")
                if len(missing_keys) <= 5:
                    print(f"[Debug] 缺失的键: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"[Debug] 意外的权重键数量: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 5:
                    print(f"[Debug] 意外的键: {unexpected_keys[:5]}")

        # 线性 probe 时冻结 encoder 参数
        if self.freeze_encoder:
            for p in self.target_encoder.parameters():
                p.requires_grad = False

        # 输入为 6 通道（Sleep数据）
        self.chan_conv       = Conv1dWithConstraint(6, self.chans_num, 1, max_norm=1)
        
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
            # 默认计算：假设patch_stride=None，seq_len=7680, patch_size=64
            seq_len = 30 * 256  # 30秒@256Hz = 7680
            patch_size = 64
            self.N = seq_len // patch_size  # 7680 // 64 = 120
            if self.debug:
                print(f"[Debug] 使用默认计算N={self.N} (seq_len={seq_len}, patch_size={patch_size})")
        
        embed_num = 4
        embed_dim = 512
        
        # 为了减少参数量，在输入分类器之前会对时间维度做平均池化
        # 将N从120降到1，这样可以大幅减少参数量
        # 分类器设计时使用N_pooled=1（pooling后的N值）
        self.N_pooled = 1  # pooling后的N值
        
        # 分类头：直接展平所有维度，然后通过Linear层
        # Layer 1: (N_pooled * embed_num * embed_dim) -> (N_pooled * embed_dim)
        # Layer 2: (N_pooled * embed_dim) -> embed_dim
        # Layer 3: embed_dim -> num_classes
        self.classifier = nn.Sequential(
            # Reshape: (B, N_pooled, embed_num, embed_dim) -> (B, N_pooled * embed_num * embed_dim)
            Rearrange('b n e d -> b (n e d)'),
            # Layer 1: (N_pooled * embed_num * embed_dim) -> (N_pooled * embed_dim)
            nn.Linear(self.N_pooled * embed_num * embed_dim, self.N_pooled * embed_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            # Layer 2: (N_pooled * embed_dim) -> embed_dim
            nn.Linear(self.N_pooled * embed_dim, embed_dim),
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
        x = x/10
        x = x - x.mean(dim=-2, keepdim=True)
        # Sleep 数据为 30s(7680)，这里统一插值到 7680 (30*256)
        # x = temporal_interpolation(x, 30*256)
        x = self.chan_conv(x)
        # 线性 probe 时保持 encoder 在 eval 模式；全微调时交给 Lightning 控制
        if self.freeze_encoder:
            self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))
        
        # z 的 shape: (B, N, embed_num, embed_dim) = (B, 120, 4, 512)
        # 为了减少参数量，对时间维度N做全局平均池化
        # (B, 120, 4, 512) -> (B, 4, 512) -> (B, 1, 4, 512)
        z_pooled = z.mean(dim=1, keepdim=True)  # (B, 120, 4, 512) -> (B, 1, 4, 512)
        
        # 传入classifier，分类器的N_pooled=1，所以输入是(B, 1, 4, 512)
        h = self.classifier(z_pooled)
        
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
        
def class_stats(folder: str, num_classes: int = 5, expected_channels: int = 6):
    """计算类别分布，用于计算类别权重（只统计expected_channels通道的文件）"""
    paths = [p for ext in ("*.pickle", "*.pkl", "*.pql")
             for p in glob.glob(os.path.join(folder, ext))]
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
    class_counts, n_tot = class_stats(train_dir, num_classes=num_classes, expected_channels=6)
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
    
    train_dataset = KaggleEEGDataset(data_root, split="train", expected_channels=6, target_channels=6)
    valid_dataset = KaggleEEGDataset(data_root, split="val", expected_channels=6, target_channels=6)

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
        test_dataset = KaggleEEGDataset(data_root, split="test", expected_channels=6, target_channels=6)
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

