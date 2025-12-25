#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================== finetune_eegpt_kaggleern.py =====================
# Fine-tunes an EEGPT-like encoder on KaggleERN windows stored as pickles:
#   {"signal": np.ndarray(C, T), "label": 0/1}
# Key stability changes:
#  - Per-channel z-score normalization + sanitize (nan/inf clamp)
#  - Class-imbalance handling (pos_weight)
#  - Warm-up: train adapter+head, then unfreeze encoder
#  - Lower LR by default, AMP off for stability (you can re-enable later)
#  - AUC metric reported with ACC/BACC
# ======================================================================

# ==== GPU SELECTION (edit here) ====
GPU_VISIBLE = "4"        # e.g., "0,1" or "4"
PRIMARY_LOCAL_INDEX = 0  # 0 = first GPU in GPU_VISIBLE list
# ===================================

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = GPU_VISIBLE  # must be set before importing torch

# ---- standard imports ----
import os
import sys
import glob
import math
import time
import pickle
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import replace as dataclass_replace
from itertools import product  # add this import near the other imports

# === NEW DATASET: 20 channels (from your screenshot) ===
use_channels_names = [
    "FP1","F7","F3","F8","FZ","FC4","FT8",
    "T3","C3","CZ","T4",
    "P7","CP3","CPZ","CP4",
    "P3","PZ","P4","T6"
]

CHANNEL_LIST = [
    'FP1', 'FPZ', 'FP2',
    'AF7', 'AF3', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'O1', 'OZ', 'O2',
]

# 2) 再建立 dict
CHANNEL_DICT = {ch.upper(): i for i, ch in enumerate(CHANNEL_LIST)}

# 3) 最後補 legacy 10-20 alias（T3/T4/T5/T6）
_ALIAS = {
    "T3": "T7",
    "T4": "T8",
    "T6": "P8",
}
for old, new in _ALIAS.items():
    CHANNEL_DICT[old.upper()] = CHANNEL_DICT[new.upper()]

# sanity check
print("[CHANNEL_DICT] n =", len(CHANNEL_DICT), "max_id =", max(CHANNEL_DICT.values()))
needed = [CHANNEL_DICT[ch.upper()] for ch in use_channels_names]
print("[use_channels_names -> ids]", dict(zip(use_channels_names, needed)))

# ===== Hyperparam sweep (paired by index; zip() will stop at the shortest list) =====
BS_LIST = [128, 64, 32]
LR_LIST = [1e-3, 4e-4, 1e-4]
WD_LIST = [1e-2, 1e-3]

def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model

# ================= Determinism =================
def make_deterministic(seed=42):
    import random
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

make_deterministic(42)

# ====== Constrained layers (Conv/Linear) ======
class _MaxNormMixin:
    def _enforce_max_norm(self, weight: torch.Tensor, max_norm: float, dim: int):
        with torch.no_grad():
            norms = weight.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-12)
            scale = (max_norm / norms).clamp(max=1.0)
            weight.mul_(scale)

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

# ====== Channel selector / projector to 19 canonical channels ======
class ChannelProjector(nn.Module):
    """
    If a name->index mapping to the 19 canonical channels is available, we SELECT/REORDER.
    Otherwise we fall back to a constrained 1x1 conv to learn a stable projection.
    """
    def __init__(self, in_ch: int, out_names: list[str], index_map: Optional[list[int]] = None, conv_max_norm: float = 2.0):
        super().__init__()
        self.out_names = out_names
        self.index_map = index_map  # list of indices in input that correspond to out_names order
        if self.index_map is None:
            # Learnable projection to 19, with max-norm like official
            self.proj = Conv1dWithConstraint(in_ch, len(out_names), kernel_size=1, bias=False, max_norm=conv_max_norm)
        else:
            self.register_buffer("sel_idx", torch.tensor(self.index_map, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        if hasattr(self, "sel_idx"):
            return x[:, self.sel_idx, :]
        else:
            return self.proj(x)


# ================= Logging / Tee =================
def setup_logging(output_dir: str, base_name: str = "train_log") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{base_name}_{timestamp}.log")

    class TeeIO(object):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data); s.flush()
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try: s.flush()
                except Exception: pass

    logfile = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = TeeIO(sys.stdout, logfile)
    sys.stderr = TeeIO(sys.stderr, logfile)

    def _excepthook(exc_type, exc, tb):
        import traceback
        print("=== Uncaught exception ===", file=sys.stderr)
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
    sys.excepthook = _excepthook

    print(f"[log] tee to {log_path}")
    return log_path


# ================= Config =================
@dataclass
class Config:
    # Preprocessed windows (train/val/test subfolders with .pickle/.pkl/.pql)
    data_root: str = "/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42"
    train_dir: str = "train"
    val_dir:   str = "val"
    test_dir:  str = "test"

    # Input hints (usually auto-detected)
    in_channels_hint: Optional[int] = None
    seq_len_in_hint:  Optional[int] = None

    # EEGPT pretrain shape expectations (will be overwritten by ckpt introspection)
    eegpt_channels: int = 58
    eegpt_seq_len:  int = 512

    # Checkpoint path (Lightning .ckpt from EEGPT pretraining)
    ckpt_path: str = "/work/HHRI-AI/UCSD_EEG/eeg_models/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"

    # Model capacity (will be overwritten by ckpt introspection)
    embed_dim: int = 512
    embed_num: int = 4
    depth: int = 1
    num_heads: int = 8
    patch_size: int = 64
    patch_stride: Optional[int] = 32  

    # Training
    num_workers: int = 4
    # batch_size: int = 32
    # epochs: int = 50
    # lr: float = 1e-3
    # weight_decay: float = 0.01
    use_amp: bool = False            # start stable; you can set True later
    grad_clip: Optional[float] = None
    probe_only: bool = True
    warmup_epochs: int = 0

    max_lr: float = 4e-4
    pct_start: float = 0.2
    epochs: int = 50
    batch_size: int = 256
    weight_decay: float = 1e-3

    # Saving
    output_dir: str = "./20251225_output/eegpt_linear_probe"
    save_best_only: bool = True

CFG = Config()


# ================= Data =================
class PickleWindowDataset(Dataset):
    """
    Expects each pickle to be a dict: {"signal": np.ndarray(C, T), "label": 0/1}.
    Loads raw (C,T), applies NaN/Inf sanitization only; all preprocessing is in-model.
    """
    def __init__(self, folder: str):
        if not os.path.isdir(folder):
            raise RuntimeError(f"Folder not found: {folder}")
        self.paths = sorted([p for ext in ("*.pickle", "*.pkl", "*.pql")
                             for p in glob.glob(os.path.join(folder, ext))])
        if len(self.paths) == 0:
            raise RuntimeError(f"No window files found in {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            obj = pickle.load(f)

        # Strict to KaggleERN format
        X = np.asarray(obj["X"], dtype=np.float32)  # (C, T)
        y = float(obj["y"])

        # Sanitize: replace NaN/Inf and clamp extreme outliers defensively
        X = np.nan_to_num(X, posinf=0.0, neginf=0.0)

        X = torch.from_numpy(X)                  # (C, T)
        y = torch.tensor([y], dtype=torch.float32)
        return X, y

# 你需要從官方 repo 複製/匯入 CHANNEL_DICT
# 例如：從 downstream/Modules/models/EEGPT_mcae.py 取得 CHANNEL_DICT
def prepare_chan_ids(channels: list[str], channel_dict: dict) -> torch.Tensor:
    chan_ids = []
    for ch in channels:
        ch = ch.upper().strip().strip(".")
        # common cleanups (optional but helpful)
        ch = ch.replace("EEG ", "").replace("EEG_", "")
        assert ch in channel_dict, f"Unknown channel name: {ch}"
        chan_ids.append(channel_dict[ch])
    return torch.tensor(chan_ids).unsqueeze(0).long()

def class_stats(folder: str):
    """Compute positive class count for pos_weight."""
    paths = [p for ext in ("*.pickle", "*.pkl", "*.pql")
             for p in glob.glob(os.path.join(folder, ext))]
    n_pos = n_tot = 0
    for p in paths:
        with open(p, "rb") as f:
            y = float(pickle.load(f)["y"])
        n_pos += int(y == 1.0)
        n_tot += 1
    return n_pos, n_tot


def make_loaders(cfg: Config):
    def _folder(name): return os.path.join(cfg.data_root, name)
    train_dir, val_dir, test_dir = _folder(cfg.train_dir), _folder(cfg.val_dir), _folder(cfg.test_dir)

    n_pos, n_tot = class_stats(train_dir)
    n_neg = max(1, n_tot - n_pos)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
    print(f"[data] train samples={n_tot}, pos={n_pos}, neg={n_neg}, pos_weight={pos_weight.item():.3f}")

    ds_train = PickleWindowDataset(train_dir)
    ds_val   = PickleWindowDataset(val_dir)
    ds_test  = PickleWindowDataset(test_dir)

    def _collate(batch):
        Xs, ys = zip(*batch)
        return torch.stack(Xs, 0), torch.stack(ys, 0)

    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    loader_val   = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    loader_test  = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    return loader_train, loader_val, loader_test, pos_weight


def peek_input_shape(loader: DataLoader) -> Tuple[int, int]:
    """Grab one batch to infer (C, T)."""
    X0, _ = next(iter(loader))
    C_in, T_in = X0.shape[1], X0.shape[2]
    return int(C_in), int(T_in)

# ===== Temporal interpolation helper (official stand-in) =====
import torch.nn.functional as F

def temporal_interpolation(x: torch.Tensor,
                           desired_sequence_length: int,
                           mode: str = "nearest",
                           use_avg: bool = True) -> torch.Tensor:
    # 官方預設：先做 across-channel mean removal
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)

    if x.ndim == 2:
        # (C, T) -> (1, C, T)
        x = x.unsqueeze(0)
        return F.interpolate(x, desired_sequence_length, mode=mode).squeeze(0)

    if x.ndim == 3:
        # (B, C, T)
        if mode in ("linear", "bilinear", "bicubic", "trilinear"):
            return F.interpolate(x, size=desired_sequence_length, mode=mode, align_corners=False)
        return F.interpolate(x, size=desired_sequence_length, mode=mode)

    raise ValueError(f"Unsupported input shape: {x.shape}")

# ================= EEGPT Backbone (Encoder only) =================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(58, 1024), patch_size=64, patch_stride=None, embed_dim=512):
        super().__init__()
        C, T = img_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = (C, T // patch_size)
        else:
            self.num_patches = (C, ((T - patch_size) // patch_stride + 1))
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, patch_size),
                              stride=(1, patch_size if patch_stride is None else patch_stride))

    def forward(self, x):
        x = x.unsqueeze(1)                # (B,1,C,T)
        x = self.proj(x).transpose(1, 3)  # (B, T_p, C, D)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob is None or self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop
        self.is_causal = is_causal
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0., is_causal=self.is_causal)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.proj(y); y = self.proj_drop(y)
        return y


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim*mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EEGTransformerEncoder(nn.Module):
    """
    Minimal EEGPT encoder, compatible with EEGPT 'encoder' weights.
    - Input: (B, C_in=58, T=1024)
    - Output: (B, N_patches, embed_num, embed_dim)
    """
    def __init__(self, img_size=(58,1024), patch_size=64, patch_stride=None,
                 embed_dim=512, embed_num=4, depth=1, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      patch_stride=patch_stride, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches  # (C, N)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        nn.init.trunc_normal_(self.summary_token, std=0.02)

        # Channel embedding (learned); per-patch positional term will be sinusoidal (computed on the fly)
        self.chan_embed = nn.Embedding(128, embed_dim)
        nn.init.trunc_normal_(self.chan_embed.weight, std=0.02)


    def forward(self, x, chan_ids):
        x = self.patch_embed(x)                     # (B, N, C, D)
        B, N, C, D = x.shape

        # add channel embedding
        x = x + self.chan_embed(chan_ids.to(x.device).long()).unsqueeze(0)  # (1,1,C,D)

        # add per-patch positional embedding (broadcast over channels)
        # shape we want to add: (1, N, 1, D)
        def _sinusoidal_pos_emb(n: int, d: int, device, dtype):
            # standard transformer sinusoidal PE
            pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)                 # (N, 1)
            i   = torch.arange(d, device=device, dtype=dtype).unsqueeze(0)                 # (1, D)
            div = torch.exp((-(2 * (i // 2)) * math.log(10000.0) / d))                     # (1, D)
            pe  = pos * div                                                                 # (N, D)
            pe[:, 0::2] = torch.sin(pe[:, 0::2])
            pe[:, 1::2] = torch.cos(pe[:, 1::2])
            return pe.view(1, n, 1, d)                                                     # (1, N, 1, D)

        x = x + _sinusoidal_pos_emb(N, D, x.device, x.dtype)

        # transformer over (channels + summary tokens), per time-patch
        x = x.flatten(0, 1)                         # (B*N, C, D)
        summary = self.summary_token.expand(x.shape[0], -1, -1)  # (B*N, embed_num, D)
        x = torch.cat([x, summary], dim=1)          # (B*N, C+embed_num, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, -self.embed_num:, :])    # keep summary tokens only → (B*N, embed_num, D)
        x = x.view(B, N, self.embed_num, D)         # (B, N, embed_num, D)
        return x

# ================= Classifier Wrapper =================
class EEGPTStressClassifier(nn.Module):
    """
    Match EEGPT official finetuning:
      - Preproc: x/10, subtract mean across channels, interpolate T->512
      - Channel 1x1 conv: in_channels -> 19 (constrained conv in paper; plain Conv1d here)
      - Encoder: (C=19, T=512), patch=64, stride=32, embed_num=4, embed_dim=512, depth=8, heads=8
      - Head: Dropout(0.5) -> Linear(2048->16) per patch, flatten (N*16), Linear(240->2)
    """
    def __init__(self, cfg: Config, detected_in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = int(detected_in_channels)

        # 1) channel conv to the 19 official channels (use a simple 1x1 conv)
        # If you can provide a mapping from your dataset channel order -> use_channels_names order,
        # pass it as `index_map`. Otherwise this will fall back to a constrained 1x1 projection.
        self.expected_channels = len(use_channels_names)
        self.chan_proj = None
        if self.in_channels != self.expected_channels:
            print(f"[warn] Input channels={self.in_channels}, expected={self.expected_channels}. "
                f"Using learnable ChannelProjector {self.in_channels}->{self.expected_channels}.")
            self.chan_proj = ChannelProjector(
                in_ch=self.in_channels,
                out_names=use_channels_names,
                index_map=None,          # 不知道 mapping，就讓它 learn
                conv_max_norm=2.0
            )

        self.encoder = EEGTransformerEncoder(
            img_size=(self.expected_channels, cfg.eegpt_seq_len),
            patch_size=cfg.patch_size,
            patch_stride=cfg.patch_stride,
            embed_dim=cfg.embed_dim,
            embed_num=cfg.embed_num,
            depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=4.0,
        )

        # 3) Official probe head: 2048 -> 16 per patch, then (N*16) -> 2
        #    N depends on T, patch_size, patch_stride. With 512/64/32 -> N=15.
        dummy_num_patches_time = ((cfg.eegpt_seq_len - cfg.patch_size) // cfg.patch_stride) + 1
        self.N = dummy_num_patches_time

        self.drop = nn.Dropout(p=0.50)
        self.linear_probe1 = LinearWithConstraint(self.cfg.embed_num * self.cfg.embed_dim, 16, bias=True, max_norm=2.0)
        self.linear_probe2 = LinearWithConstraint(self.N * 16, 2, bias=True, max_norm=2.0)

        # Channel IDs (we don't have prepare_chan_ids here; use a simple range)
        self.register_buffer("chan_ids", prepare_chan_ids(use_channels_names, CHANNEL_DICT))

    def _interp_to_512(self, x: torch.Tensor) -> torch.Tensor:
        # shape: (B, C, T) -> interpolate to T=512 using linear mode
        T_tar = self.cfg.eegpt_seq_len
        if x.shape[-1] == T_tar:
            return x
        return temporal_interpolation(x, T_tar)


    def forward(self, x):
        # Preproc to match official code more closely:
        # 1) per-channel (over time) z-score
        # Official-like: scale, subtract mean across channels (dim=-2), no per-channel z-score
        x = x / 10.0
        x = torch.nan_to_num(x, posinf=0.0, neginf=0.0)

        if self.chan_proj is not None:
            x = self.chan_proj(x)  # (B, 30, T) -> (B, 19, T)

        x = temporal_interpolation(x, self.cfg.eegpt_seq_len, mode="nearest", use_avg=True)

        assert x.shape[1] == self.chan_ids.shape[1], \
            f"Channel mismatch: x has {x.shape[1]} ch, chan_ids has {self.chan_ids.shape[1]} ch"

        # Encoder + probe head
        z = self.encoder(x, self.chan_ids)        # (B, N, E, D)
        h = z.flatten(2)                                     # (B, N, 2048)
        h = self.linear_probe1(self.drop(h))                 # (B, N, 16)
        h = h.flatten(1)                                     # (B, N*16) == (B, 240)
        logits = self.linear_probe2(h)                       # (B, 2)
        return logits


# ================= Checkpoint Loading (Torch 2.6-safe) =================
def _safe_load_ckpt(ckpt_path: str):
    """Robust loader for PyTorch 2.6 and older checkpoints."""
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")
    except pickle.UnpicklingError:
        from torch.serialization import add_safe_globals
        add_safe_globals([getattr])
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def inspect_eegpt_ckpt_meta(ckpt_path: str):
    """
    Peek EEGPT encoder shapes from a Lightning .ckpt and return:
      chan_n (from chan_embed), embed_dim (model width), embed_num (summary tokens)
    """
    obj = _safe_load_ckpt(ckpt_path)
    sd = obj.get("state_dict", obj)

    def pick(*keys):
        for k in keys:
            if k in sd:
                return sd[k]
        return None

    # channel embeddings
    ce = pick("encoder.chan_embed.weight",
              "model.encoder.chan_embed.weight",
              "encoder.module.encoder.chan_embed.weight")
    chan_n = int(ce.shape[0]) if ce is not None else 58

    # summary token => (1, embed_num, embed_dim)
    st = pick("encoder.summary_token",
              "model.encoder.summary_token",
              "encoder.module.encoder.summary_token")
    if st is not None and st.ndim == 3:
        _, embed_num, embed_dim = st.shape
    else:
        pe = pick("encoder.patch_embed.proj.weight",
                  "model.encoder.patch_embed.proj.weight")
        embed_dim = int(pe.shape[0]) if pe is not None else 512
        embed_num = 1

    return chan_n, embed_dim, embed_num


def load_eegpt_encoder_from_ckpt(model: nn.Module, ckpt_path: str):
    """
    Loads only the EEGPT encoder weights from a Lightning .ckpt into model.encoder.
    Also resizes chan_embed to match ckpt if needed.
    """
    ckpt = _safe_load_ckpt(ckpt_path)
    state = ckpt.get("state_dict", ckpt)

    # Extract encoder substate
    enc_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            enc_state[k[len("encoder."):]] = v
        elif k.startswith("model.encoder."):
            enc_state[k[len("model.encoder."):]] = v
        elif k.startswith("target_encoder."):  # <-- add this
            enc_state[k[len("target_encoder."):]] = v

    target_encoder = model.encoder  # no DataParallel to keep things simple

    # Handle chan_embed shape mismatch
    ce_key = "chan_embed.weight"
    if ce_key in enc_state:
        ckpt_embed = enc_state[ce_key]
        ckpt_n, ckpt_dim = ckpt_embed.shape
        curr_n, curr_dim = target_encoder.chan_embed.weight.shape
        if (ckpt_n != curr_n) or (ckpt_dim != curr_dim):
            print(f"[EEGPT load] Resizing chan_embed: model {curr_n}x{curr_dim} -> ckpt {ckpt_n}x{ckpt_dim}")
            device = target_encoder.chan_embed.weight.device
            dtype  = target_encoder.chan_embed.weight.dtype
            new_emb = nn.Embedding(ckpt_n, ckpt_dim).to(device=device, dtype=dtype)
            with torch.no_grad():
                new_emb.weight.copy_(ckpt_embed.to(device=device, dtype=dtype))
            target_encoder.chan_embed = new_emb

    missing, unexpected = target_encoder.load_state_dict(enc_state, strict=False)
    print("[EEGPT load] Missing keys:", missing)
    print("[EEGPT load] Unexpected keys:", unexpected)


# ================= Evaluation =================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_prob.append(probs)
        y_true.append(y.squeeze(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    acc = float((tp + tn) / max(1, len(y_true)))
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    bacc = float(0.5 * (tpr + tnr))

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {"acc": acc, "bacc": bacc, "auc": auc}


# ================= Main =================
def main():
    cfg_master = CFG  # keep a master copy to reset per run

    # iterate over paired combos (zip stops at shortest list)
    for idx, (bs, lr, wd) in enumerate(product(BS_LIST, LR_LIST, WD_LIST), start=1):
        # ----- per-run config -----
        cfg = dataclass_replace(cfg_master, batch_size=bs, max_lr=lr, weight_decay=wd)

        # make a clean per-run folder name
        def _fmt(x):
            s = f"{x:g}" if isinstance(x, float) else str(x)
            return s.replace(".", "p")
        run_name = f"bs{_fmt(bs)}_lr{_fmt(lr)}_wd{_fmt(wd)}"
        run_dir = os.path.join(cfg.output_dir, run_name)

        os.makedirs(run_dir, exist_ok=True)
        log_path = setup_logging(run_dir, base_name="train_log")
        print(f"[sweep] run {idx}: bs={bs}, lr={lr}, wd={wd}")
        print(f"[sweep] output dir: {run_dir}")

        # Device selection (reprint per run for clarity)
        device = torch.device(f"cuda:{PRIMARY_LOCAL_INDEX}" if torch.cuda.is_available() else "cpu")
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("torch.cuda.device_count() =", torch.cuda.device_count())
        print("Using device:", device)
        print("Log file:", log_path)

        # --- Data (per run, in case batch size changed etc.) ---
        train_loader, val_loader, test_loader, pos_weight = make_loaders(cfg)

        # Peek input shape
        detected_C, detected_T = peek_input_shape(train_loader)
        if cfg.in_channels_hint is not None and cfg.in_channels_hint != detected_C:
            print(f"[warn] in_channels_hint={cfg.in_channels_hint} differs from detected {detected_C}; using detected.")
        if cfg.seq_len_in_hint is not None and cfg.seq_len_in_hint != detected_T:
            print(f"[warn] seq_len_in_hint={cfg.seq_len_in_hint} differs from detected {detected_T}; using detected.")
        print(f"[shape] detected input: C={detected_C}, T={detected_T}")

        # Inspect EEGPT ckpt to discover its hyperparams
        assert os.path.isfile(cfg.ckpt_path), f"Checkpoint not found: {cfg.ckpt_path}"
        ckpt_chan_n, ckpt_embed_dim, ckpt_embed_num = inspect_eegpt_ckpt_meta(cfg.ckpt_path)
        print(f"[ckpt] encoder meta -> chan_n={ckpt_chan_n}, embed_dim={ckpt_embed_dim}, embed_num={ckpt_embed_num}")

        # Match config to ckpt
        cfg.eegpt_channels = ckpt_chan_n
        cfg.embed_dim      = ckpt_embed_dim
        cfg.embed_num      = ckpt_embed_num

        # ----- Build+wrap model -----
        model = EEGPTStressClassifier(cfg, detected_in_channels=detected_C).to(device)
        load_eegpt_encoder_from_ckpt(model, cfg.ckpt_path)

        assert int(model.chan_ids.max()) < model.encoder.chan_embed.num_embeddings, \
        f"chan_ids out of range: max={int(model.chan_ids.max())}, emb_n={model.encoder.chan_embed.num_embeddings}"

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"[gpu] Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        base = _unwrap(model)
        # ----- Criterion/optimizer/scaler -----
        w_pos = float(pos_weight.item())
        class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Warm-up: freeze encoder
        # Select trainable params based on mode
        if cfg.probe_only:
            # 只訓練 probe head（因為你已經移除 chan_conv）
            trainable_params = list(base.linear_probe1.parameters()) \
                            + list(base.linear_probe2.parameters())
            for p in base.encoder.parameters():
                p.requires_grad = False
            base.encoder.eval()

        else:
            # Full fine-tune: everything
            for p in base.encoder.parameters():
                p.requires_grad = True
            base.encoder.train()
            trainable_params = base.parameters()

        optimizer = optim.AdamW(trainable_params, lr=cfg.max_lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            epochs=cfg.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=cfg.pct_start,
            anneal_strategy="cos"
        )

        scaler = torch.amp.GradScaler(
            device="cuda" if device.type == "cuda" else "cpu",
            enabled=(cfg.use_amp and device.type == "cuda")
        )

        best_val = -1.0

        # ----- Training loop -----
        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            t0 = time.time()

            for X, y in train_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=(cfg.use_amp and device.type == "cuda")
                ):
                    logits = model(X)
                    loss = criterion(logits, y.squeeze(1).long())

                if not torch.isfinite(loss):
                    print("[warn] non-finite loss detected; skipping batch")
                    continue

                scaler.scale(loss).backward()
                if cfg.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(base.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # --- Max-norm constraints to mimic paper ---
                base = _unwrap(model)
                
                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

            dur = time.time() - t0
            train_loss = epoch_loss / max(1, n_batches)
            val_metrics = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1:03d}/{cfg.epochs} | loss={train_loss:.4f} | "
                  f"val_acc={val_metrics['acc']:.4f} | val_bacc={val_metrics['bacc']:.4f} | "
                  f"val_auc={val_metrics['auc']:.4f} | {dur:.1f}s")

            # Save best (by bacc) into this run_dir
            score = val_metrics['bacc']
            state_to_save = _unwrap(model).state_dict()
            to_save = {
                "epoch": epoch,
                "model": state_to_save,
                "optim": optimizer.state_dict(),
                "cfg": cfg.__dict__,
                "best_val": best_val
            }
            if cfg.save_best_only:
                if score > best_val:
                    best_val = score
                    to_save["best_val"] = best_val
                    torch.save(to_save, os.path.join(run_dir, "checkpoint-best.pth"))
                    print(f"  -> Saved BEST (bacc={best_val:.4f})")
            else:
                torch.save(to_save, os.path.join(run_dir, f"checkpoint-epoch{epoch+1:03d}.pth"))

            # Always save rolling last for safety
            torch.save(to_save, os.path.join(run_dir, "checkpoint-last.pth"))

        # ----- Test (best) -----
        best_path = os.path.join(run_dir, "checkpoint-best.pth") if cfg.save_best_only else None
        if best_path and os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location="cpu")
            _unwrap(model).load_state_dict(ckpt["model"], strict=True)
            print(f"Loaded best: {best_path}")

        test_metrics = evaluate(model, test_loader, device)
        print("========== TEST ==========")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")
        print("==========================")

        # ----- Export per-run -----
        export_path = os.path.join(run_dir, "eegpt_stress_finetuned.pth")
        torch.save(_unwrap(model).state_dict(), export_path)
        print(f"Exported fine-tuned weights to: {export_path}")


if __name__ == "__main__":
    main()
