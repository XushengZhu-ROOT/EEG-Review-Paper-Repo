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
from sklearn.metrics import confusion_matrix
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
# 当前数据（可以根据自己的数据集修改）：
# - 二分类（0/1）
# - 3 秒，采样率 256 Hz -> 时间长度 768
# - 56 通道，通道名如下（需与实际数据顺序保持一致）

use_channels_names = [
    'FP1', 'FP2', 'AF7', 'AF3', 'AF4',
    'AF8', 'F7', 'F5', 'F3', 'F1',
    'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ',
    'FC2', 'FC4', 'FC6', 'FT8', 'T7',
    'C5', 'C3', 'C1', 'CZ', 'C2',
    'C4', 'C6', 'T8', 'TP7', 'CP5',
    'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'CP6', 'TP8', 'P7', 'P5', 'P3',
    'P1', 'PZ', 'P2', 'P4', 'P6',
    'P8', 'PO7', 'POZ', 'PO8', 'O1',
    'O2',
]

# ==================== 用户可配置参数 ====================
# 建议你只改这一段就能完成大部分自定义

# 任务名称（会用于输出文件夹命名）
TASK_NAME = "EEGPT_KaggleERN"

# 数据路径（kaggle_data 目录位置）
data_root = "./kaggle_data"

# 分类类别数
num_classes = 2

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
BS_LIST = [6]#[128, 64, 32]
LR_LIST = [1e-3, 4e-4, 1e-4]
WD_LIST = [1e-2, 1e-3]


class KaggleEEGDataset(torch.utils.data.Dataset):
    """
    从 kaggle_data/{split}/*.pickle 读取数据。
    每个 pickle 是一个 dict:
        - 'signal': ndarray, shape (56, 768)
        - 'label' : int, 0/1
        - 'epoch_id': str
    """
    def __init__(self, root_dir: str, split: str = "train"):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        assert os.path.isdir(self.split_dir), f"{self.split_dir} 不存在"

        self.files = [
            os.path.join(self.split_dir, f)
            for f in os.listdir(self.split_dir)
            if f.endswith(".pickle")
        ]
        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f"{self.split_dir} 下没有找到任何 .pickle 文件")

        print(f"[KaggleEEGDataset] split={split}, 样本数={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)

        x = obj["signal"]   # (56, 768)
        y = obj["label"]    # int 0/1

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
        self.chans_num = len(use_channels_names)
        self.num_class = num_classes
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_ratio = encoder_lr_ratio  # encoder 学习率相对于 head 的比例
        # test dataloader（在 main 里赋值，用于每个 epoch 做 test）
        self.test_loader = None

        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 3 * 256],  # 56 通道, 3s@256Hz = 768
            patch_size=32*2,
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

        # 输入为 56 通道（与你的数据一致，如不同需修改这里）
        self.chan_conv       = Conv1dWithConstraint(56, self.chans_num, 1, max_norm=1)
        
        # 新的分类头结构（参考数据集信息和实验笔记，但做了参数量裁剪）
        # z 的 shape: (B, N, embed_num, embed_dim)
        # 这里只在 N 维（patch 维）上做平均池化，保留 embed_num 这一维：
        #   (B, N, 4, 512) --mean over N--> (B, 4, 512)
        # 然后展平成 (B, 4*512=2048)，再做 2048 -> 512 -> 512 -> num_classes
        embed_num = 4
        embed_dim = 512
        in_dim = embed_num * embed_dim      # 2048
        hidden_dim = embed_dim             # 512

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_class),
        )
       
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[]}
        self.is_sanity=True
        
        # 用于记录混淆矩阵和指标
        self.output_dir = None  # 将在训练时设置
        
    
    def forward(self, x):
        B, C, T = x.shape
        x = x/100
        x = x - x.mean(dim=-2, keepdim=True)
        # 你的数据为 3s(768)，这里统一插值到 768；如需裁剪到 2s，可改为 2*256
        x = temporal_interpolation(x, 3*256)
        x = self.chan_conv(x)
        # 线性 probe 时保持 encoder 在 eval 模式；全微调时交给 Lightning 控制
        if self.freeze_encoder:
            self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))
        
        # z 的 shape: (B, N, embed_num, embed_dim)
        # 先在 N 维（patch 维）做平均池化，保留 embed_num 维：
        # (B, N, 4, 512) -> (B, 4, 512)
        if z.dim() == 4:
            z = z.mean(dim=1)
        elif z.dim() > 4:
            # 兼容性：如果维度更多，先把多余维度平均掉，再按 (B, N, E, D) 理解
            while z.dim() > 4:
                z = z.mean(dim=2)
            z = z.mean(dim=1)
        else:
            # 理论上不会走到这里，只是防御
            z = z.mean(dim=1)

        # (B, 4, 512) -> (B, 4*512)
        z = z.flatten(start_dim=1)

        h = self.classifier(z)
        
        return x, h

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        y_score =  logit.clone().detach().cpu()
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # rocauc = metrics.roc_auc_score(label.clone().detach().cpu(), y_score)
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
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        
        # 计算预测类别（用于混淆矩阵）
        y_pred = (y_score > 0.5).long().numpy()
        label_np = label.numpy()
        
        # 计算指标
        metric_list = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.numpy(), label_np, metric_list, True)
        
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
                    y_score_test = torch.softmax(logits, dim=-1)[:, 1]
                    all_labels.append(y.cpu())
                    all_scores.append(y_score_test.cpu())

            labels_t = torch.cat(all_labels, dim=0).numpy()
            scores_t = torch.cat(all_scores, dim=0).numpy()
            preds_t = (scores_t > 0.5).astype(int)

            metric_list_t = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
            test_metrics = get_metrics(scores_t, labels_t, metric_list_t, True)
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
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        
        # 保存logits用于epoch结束时的评估
        y_score = torch.softmax(logit, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

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
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        
        # 计算训练集指标
        y_pred = (y_score > 0.5).long().numpy()
        label_np = label.numpy()
        
        metric_list = ["accuracy", "balanced_accuracy"]
        results = get_metrics(y_score.numpy(), label_np, metric_list, True)
        
        rocauc = metrics.roc_auc_score(label_np, y_score.numpy())
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False)
        
        # 打印训练指标
        epoch = self.current_epoch
        acc = results.get('accuracy', 0)
        bacc = results.get('balanced_accuracy', 0)
        print(f"[Epoch {epoch}] Train - Acc: {acc:.4f}, BAcc: {bacc:.4f}, ROC-AUC: {rocauc:.4f}")
        
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
        
def main():
    # 重新设定种子（实验级别）
    seed_torch(9)
    # 创建输出文件夹（每次运行一个实验，一个文件夹）
    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{TASK_NAME}_bs{batch_size}_lr{max_lr}_wd{weight_decay}"
        f"_freeze{int(freeze_encoder)}_{run_tag}"
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
        "num_classes": 2,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ==================== 数据加载 ====================
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


if __name__ == "__main__":
    # 网格搜索：遍历所有 (batch_size, lr, weight_decay) 组合
    for bs in BS_LIST:
        for lr in LR_LIST:
            for wd in WD_LIST:
                print("\n" + "=" * 80)
                print(f"开始新实验: bs={bs}, lr={lr}, wd={wd}, freeze_encoder={freeze_encoder}")
                print("=" * 80)

                # 更新全局超参数
                batch_size = bs
                max_lr = lr
                weight_decay = wd

                # 运行一次完整实验（会自动创建带超参信息的输出文件夹）
                main()

