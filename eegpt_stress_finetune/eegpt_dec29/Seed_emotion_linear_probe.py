#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================== finetune_eegpt_seed_emotion.py =====================
# Fine-tunes an EEGPT-like encoder on SEED emotion 7-class windows stored as pickles:
#   {"signal": np.ndarray(62, T), "label": 0-6}
# Key stability changes:
#  - Per-channel z-score normalization + sanitize (nan/inf clamp)
#  - Class-imbalance handling (class weights)
#  - Warm-up: train adapter+head, then unfreeze encoder
#  - Lower LR by default, AMP off for stability (you can re-enable later)
#  - Multi-class metrics: ACC, per-class accuracy, confusion matrix
# ======================================================================

# ==== GPU SELECTION (edit here) ====
GPU_VISIBLE = "0"        # e.g., "0,1" or "4"
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
from einops.layers.torch import Rearrange

# SEED emotion dataset: 60 channels (removed CB1 and CB2 as they are not in CHANNEL_DICT)
# Original 62-channel order: ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
#                             'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
#                             'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
#                             'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#                             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
#                             'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
#                             'CB1', 'O1', 'OZ', 'O2', 'CB2']
# CB1 is at index 57, CB2 is at index 61 (0-based)
use_channels_names = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'O1', 'OZ', 'O2'
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

CHANNEL_DICT = {ch.upper(): i for i, ch in enumerate(CHANNEL_LIST)}

# sanity check
print("[CHANNEL_DICT] n =", len(CHANNEL_DICT), "max_id =", max(CHANNEL_DICT.values()))
needed = [CHANNEL_DICT[ch.upper()] for ch in use_channels_names]
print("[use_channels_names -> ids]", dict(zip(use_channels_names, needed)))

# ===== Hyperparam sweep (paired by index; zip() will stop at the shortest list) =====
BS_LIST = [64]
LR_LIST = [4e-4]
WD_LIST = [1e-2]

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
    data_root: str = "./seed_data"
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
    ckpt_path: str = "./eegpt_mcae_58chs_4s_large4E.ckpt"

    # Model capacity (will be overwritten by ckpt introspection)
    embed_dim: int = 512
    embed_num: int = 4
    depth: int = 8  # Pre-trained model uses depth=8 (blocks.0 to blocks.7)
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
    output_dir: str = "./seed_output/eegpt_linear_probe"
    save_best_only: bool = True

CFG = Config()


# ================= Data =================
class PickleWindowDataset(Dataset):
    """
    Expects each pickle to be a dict: {"signal": np.ndarray(C, T), "label": 0-6}.
    Loads raw (C,T), applies NaN/Inf sanitization only; all preprocessing is in-model.
    - Accepts 62-channel data, removes CB1 and CB2 (indices 57 and 61) to get 60 channels
    - Supports subdirectories (e.g., train/subject_X/)
    """
    def __init__(self, folder: str, expected_channels: int = 62, target_channels: int = 60):
        if not os.path.isdir(folder):
            raise RuntimeError(f"Folder not found: {folder}")
        # Recursively find all pickle files in subdirectories
        all_paths = sorted([p for ext in ("*.pickle", "*.pkl", "*.pql")
                           for p in glob.glob(os.path.join(folder, "**", ext), recursive=True)])
        
        # Filter paths: only keep files with expected_channels
        self.paths = []
        self.expected_channels = expected_channels
        self.target_channels = target_channels
        # CB1 and CB2 indices in 62-channel data (0-based)
        self.cb_indices = [57, 61]  # CB1 at 57, CB2 at 61
        skipped = 0
        for p in all_paths:
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                X = np.asarray(obj["signal"], dtype=np.float32)
                if X.shape[0] == expected_channels:
                    self.paths.append(p)
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                continue
        
        if len(self.paths) == 0:
            raise RuntimeError(f"No valid {expected_channels}-channel window files found in {folder}")
        if skipped > 0:
            print(f"[dataset] Skipped {skipped} files with != {expected_channels} channels")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            obj = pickle.load(f)

        # SEED emotion 7-class format
        X = np.asarray(obj["signal"], dtype=np.float32)  # (C, T)
        y = int(obj["label"])  # 0-6 for 7 classes

        # Check channel count (should be expected_channels)
        if X.shape[0] != self.expected_channels:
            raise ValueError(f"Expected {self.expected_channels} channels, got {X.shape[0]}")

        # Remove CB1 and CB2 channels (indices 57 and 61) to get 60 channels
        # Create mask to keep all channels except CB1 and CB2
        keep_indices = [i for i in range(X.shape[0]) if i not in self.cb_indices]
        X = X[keep_indices, :]  # (60, T)

        # Sanitize: replace NaN/Inf and clamp extreme outliers defensively
        X = np.nan_to_num(X, posinf=0.0, neginf=0.0)

        X = torch.from_numpy(X)                  # (60, T)
        y = torch.tensor(y, dtype=torch.long)    # Long tensor for classification
        return X, y

# 你需要從官方 repo 複製/匯入 CHANNEL_DICT
# 例如：從 downstream/Modules/models/EEGPT_mcae.py 取得 CHANNEL_DICT

def prepare_chan_ids(channels: list[str], channel_dict: dict) -> torch.Tensor:
    chan_ids = []
    for ch in channels:
        ch = ch.upper().strip(".")
        assert ch in channel_dict, f"Unknown channel name: {ch}"
        chan_ids.append(channel_dict[ch])
    return torch.tensor(chan_ids).unsqueeze(0).long()

def class_stats(folder: str, num_classes: int = 7, expected_channels: int = 62):
    """Compute class distribution for class weights (only for expected_channels)."""
    paths = [p for ext in ("*.pickle", "*.pkl", "*.pql")
             for p in glob.glob(os.path.join(folder, "**", ext), recursive=True)]
    class_counts = [0] * num_classes
    n_tot = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            X = np.asarray(obj["signal"], dtype=np.float32)
            # Only count files with expected_channels
            if X.shape[0] == expected_channels:
                y = int(obj["label"])
                if 0 <= y < num_classes:
                    class_counts[y] += 1
                    n_tot += 1
        except Exception:
            continue
    return class_counts, n_tot


def make_loaders(cfg: Config, num_classes: int = 7):
    def _folder(name): return os.path.join(cfg.data_root, name)
    train_dir, val_dir, test_dir = _folder(cfg.train_dir), _folder(cfg.val_dir), _folder(cfg.test_dir)

    class_counts, n_tot = class_stats(train_dir, num_classes=num_classes, expected_channels=62)
    # Compute class weights: inverse frequency weighting
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(n_tot / (num_classes * count))
        else:
            class_weights.append(1.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"[data] train samples={n_tot}, class_counts={class_counts}")
    print(f"[data] class_weights={class_weights.tolist()}")

    ds_train = PickleWindowDataset(train_dir, expected_channels=62, target_channels=60)
    ds_val   = PickleWindowDataset(val_dir, expected_channels=62, target_channels=60)
    ds_test  = PickleWindowDataset(test_dir, expected_channels=62, target_channels=60)

    def _collate(batch):
        Xs, ys = zip(*batch)
        return torch.stack(Xs, 0), torch.stack(ys, 0)

    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    loader_val   = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    loader_test  = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    return loader_train, loader_val, loader_test, class_weights


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
                 embed_dim=512, embed_num=4, depth=3, num_heads=8, mlp_ratio=4.0):
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
    Match EEGPT official finetuning for SEED emotion 7-class task:
      - Preproc: x/10, subtract mean across channels, interpolate T->512
      - Channel 1x1 conv: in_channels -> 60 (removed CB1 and CB2, constrained conv in paper; plain Conv1d here)
      - Encoder: (C=60, T=512), patch=64, stride=32, embed_num=4, embed_dim=512, depth=8, heads=8
      - MLP Head: Layer1(N×embed_num×embed_dim->N×embed_dim), Layer2(N×embed_dim->embed_dim), Layer3(embed_dim->7)
    """
    def __init__(self, cfg: Config, detected_in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = int(detected_in_channels)

        # 1) channel conv to the 62 SEED channels (use a simple 1x1 conv)
        # If you can provide a mapping from your dataset channel order -> use_channels_names order,
        # pass it as `index_map`. Otherwise this will fall back to a constrained 1x1 projection.
        self.chan_conv = Conv1dWithConstraint(
            in_channels=self.in_channels,
            out_channels=len(use_channels_names),
            kernel_size=1, bias=False, max_norm=2.0
        )



        # 2) EEGPT encoder (use 60x512 view, but keep summary tokens & widths from ckpt)
        self.encoder = EEGTransformerEncoder(
            img_size=(len(use_channels_names), cfg.eegpt_seq_len),   # (60, 512)
            patch_size=cfg.patch_size,                               # 64
            patch_stride=cfg.patch_stride,                           # 32
            embed_dim=cfg.embed_dim,                                 # 512
            embed_num=cfg.embed_num,                                 # 4
            depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=4.0,
        )

        # 3) MLP classifier head: 
        #    Layer 1: (N × embed_num × embed_dim) -> (N × embed_dim)
        #    Layer 2: (N × embed_dim) -> embed_dim
        #    Layer 3: embed_dim -> num_classes (7)
        dummy_num_patches_time = ((cfg.eegpt_seq_len - cfg.patch_size) // cfg.patch_stride) + 1
        self.N = dummy_num_patches_time

        self.classifier = nn.Sequential(
            # Reshape: (B, N, embed_num, embed_dim) -> (B, N * embed_num * embed_dim)
            Rearrange('b n e d -> b (n e d)'),
            # Layer 1: (N * embed_num * embed_dim) -> (N * embed_dim)
            nn.Linear(self.N * self.cfg.embed_num * self.cfg.embed_dim, self.N * self.cfg.embed_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            # Layer 2: (N * embed_dim) -> embed_dim
            nn.Linear(self.N * self.cfg.embed_dim, self.cfg.embed_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            # Layer 3: embed_dim -> num_classes
            nn.Linear(self.cfg.embed_dim, 7)  # 7 classes for emotion task
        )

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
        x = temporal_interpolation(x, self.cfg.eegpt_seq_len, mode="nearest", use_avg=True)

        # map to 19 canonical channels via constrained 1x1 conv
        x = self.chan_conv(x)

        # Encoder + classifier head
        z = self.encoder(x, self.chan_ids)        # (B, N, embed_num, embed_dim)
        logits = self.classifier(z)               # (B, 6)
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
def evaluate(model, loader, device, num_classes: int = 7):
    model.eval()
    y_true, y_pred_all = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred_all.append(y_pred)
        y_true.append(y.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred_all = np.concatenate(y_pred_all)

    # Overall accuracy
    acc = float((y_true == y_pred_all).sum() / max(1, len(y_true)))

    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = (y_true == c)
        if mask.sum() > 0:
            per_class_acc.append(float((y_pred_all[mask] == c).sum() / mask.sum()))
        else:
            per_class_acc.append(0.0)
    
    # Balanced accuracy (average of per-class accuracies)
    bacc = float(np.mean(per_class_acc))

    # Confusion matrix (optional, for detailed analysis)
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred_all, labels=list(range(num_classes)))
    except Exception:
        cm = None

    return {
        "acc": acc,
        "bacc": bacc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm
    }


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
        train_loader, val_loader, test_loader, class_weights = make_loaders(cfg, num_classes=7)

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
        class_weights_device = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_device)

        # Warm-up: freeze encoder
        # Select trainable params based on mode
        if cfg.probe_only:
            # Official behavior: train only chan_conv + classifier head
            trainable_params = list(base.chan_conv.parameters()) \
                            + list(base.classifier.parameters())
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

        scaler = torch.cuda.amp.GradScaler(
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
                    loss = criterion(logits, y.long())

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
            val_metrics = evaluate(model, val_loader, device, num_classes=7)
            per_class_str = ", ".join([f"c{i}:{val_metrics['per_class_acc'][i]:.3f}" for i in range(7)])
            print(f"Epoch {epoch+1:03d}/{cfg.epochs} | loss={train_loss:.4f} | "
                  f"val_acc={val_metrics['acc']:.4f} | val_bacc={val_metrics['bacc']:.4f} | "
                  f"per_class=[{per_class_str}] | {dur:.1f}s")

            # Track best (by bacc) - no saving to save disk space
            score = val_metrics['bacc']
            if score > best_val:
                best_val = score
                print(f"  -> New BEST (bacc={best_val:.4f})")

        # ----- Test (using final model, no checkpoint loading to save disk space) -----
        print(f"[test] Using final model (best_val_bacc={best_val:.4f})")
        test_metrics = evaluate(model, test_loader, device, num_classes=7)
        print("========== TEST ==========")
        print(f"acc: {test_metrics['acc']:.4f}")
        print(f"bacc: {test_metrics['bacc']:.4f}")
        print(f"per_class_acc: {test_metrics['per_class_acc']}")
        if test_metrics['confusion_matrix'] is not None:
            print(f"confusion_matrix:\n{test_metrics['confusion_matrix']}")
        print("==========================")

        # ----- No weight export to save disk space -----


if __name__ == "__main__":
    main()
