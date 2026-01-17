#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================== finetune_eegpt_stress.py =====================
# Multi-GPU selection INSIDE the script + tee logs to file.
# Fine-tunes EEGPT encoder (from Lightning .ckpt) on stress dataset windows.
# - PyTorch 2.6-safe torch.load()
# - Auto-detect input channels/length to build the adapter (fixes 16!=30).
# - Auto-match encoder width/summary tokens/channels to CKPT (fixes size mismatch).
# - nn.DataParallel over selected GPUs.
# ====================================================================

# ==== GPU SELECTION (edit here) ====
GPU_VISIBLE = "3,4,5"   # e.g., "0,1,2,5"
PRIMARY_LOCAL_INDEX = 0   # 0 = first GPU in GPU_VISIBLE list
# ===================================

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = GPU_VISIBLE  # must be set before importing torch

# ---- regular imports (torch AFTER env) ----
import os
import sys
import math
import time
import glob
import pickle
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

# ======= CONFIG =======
@dataclass
class Config:
    # Your preprocessed windows (train/val/test subfolders with .pickle/.pkl/.pql)
    data_root: str = "./Stress_noleak_30chan_no400up_seed_siwen42"
    train_dir: str = "train"
    val_dir:   str = "val"
    test_dir:  str = "test"

    # Set these if you want to force a shape; otherwise they will be auto-detected
    in_channels_hint: Optional[int] = None   # e.g., 30; leave None to auto-detect
    seq_len_in_hint:  Optional[int] = None   # e.g., 1000; leave None to auto-detect

    # EEGPT pretrain shape expectations (will be overwritten by ckpt introspection)
    eegpt_channels: int = 58
    eegpt_seq_len:  int = 1024

    # Checkpoint path (Lightning .ckpt from EEGPT pretraining)
    ckpt_path: str = "./eegpt_mcae_58chs_4s_large4E.ckpt"

    # Model capacity (will be overwritten by ckpt introspection)
    embed_dim: int = 512
    embed_num: int = 4
    depth: int = 8
    num_heads: int = 8
    patch_size: int = 64
    patch_stride: Optional[int] = None  # None to mimic pretrain

    # Training
    num_workers: int = 3
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-5
    weight_decay: float = 0.05
    use_amp: bool = True
    grad_clip: Optional[float] = 1.0

    # Saving
    output_dir: str = "./outputs_eegpt_stress"
    save_best_only: bool = True

CFG = Config()

# ======= DATA =======
class PickleWindowDataset(Dataset):
    def __init__(self, folder: str):
        if not os.path.isdir(folder):
            raise RuntimeError(f"Folder not found: {folder}")
        pats = []
        for ext in ("*.pickle", "*.pkl", "*.pql"):
            pats.extend(glob.glob(os.path.join(folder, ext)))
        self.paths = sorted(pats)
        if len(self.paths) == 0:
            raise RuntimeError(f"No window files found in {folder} (expected .pickle/.pkl/.pql)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            obj = pickle.load(f)
        X = obj["X"]  # (C, T)
        y = obj["y"]
        X = torch.from_numpy(np.asarray(X)).float()
        y = torch.tensor([float(y)], dtype=torch.float32)  # (1,)
        return X, y

def make_loaders(cfg: Config):
    def _folder(name): return os.path.join(cfg.data_root, name)
    ds_train = PickleWindowDataset(_folder(cfg.train_dir))
    ds_val   = PickleWindowDataset(_folder(cfg.val_dir))
    ds_test  = PickleWindowDataset(_folder(cfg.test_dir))

    def _collate(batch):
        Xs, ys = zip(*batch)
        X = torch.stack(Xs, dim=0)  # (B, C, T)
        y = torch.stack(ys, dim=0)  # (B, 1)
        return X, y

    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    loader_val   = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    loader_test  = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=_collate)
    return loader_train, loader_val, loader_test

def peek_input_shape(loader: DataLoader) -> Tuple[int, int]:
    """Grab one batch to infer (C, T)."""
    X0, _ = next(iter(loader))
    C_in, T_in = X0.shape[1], X0.shape[2]
    return int(C_in), int(T_in)

# ======= EEGPT BACKBONE (Encoder only) =======
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
    def __init__(self, drop_prob=None): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob is None or self.drop_prob == 0. or not self.training: return x
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
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x); return x

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
                 embed_dim=512, embed_num=4, depth=8, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      patch_stride=patch_stride, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches  # (C, N)

        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        nn.init.trunc_normal_(self.summary_token, std=0.02)

        # Channel embedding; rebuilt to ckpt size during loading if needed.
        self.chan_embed = nn.Embedding(128, embed_dim)

    def forward(self, x, chan_ids):
        x = self.patch_embed(x)                             # (B, N, C, D)
        B, N, C, D = x.shape
        x = x + self.chan_embed(chan_ids.to(x.device).long()).unsqueeze(0)  # (1,1,C,D)
        x = x.flatten(0, 1)                                 # (B*N, C, D)
        summary = self.summary_token.repeat(x.shape[0], 1, 1)               # (B*N, embed_num, D)
        x = torch.cat([x, summary], dim=1)                  # (B*N, C+embed_num, D)
        for blk in self.blocks:
            x = blk(x)
        x = x[:, -self.embed_num:, :]                       # keep summary tokens only
        x = self.norm(x)                                    # (B*N, embed_num, D)
        x = x.flatten(-2)                                   # (B*N, embed_num*D)
        x = x.view(B, N, self.embed_num, self.embed_dim)    # (B, N, E, D)
        return x

# ======= CLASSIFIER WRAPPER =======
class EEGPTStressClassifier(nn.Module):
    """
    - ChannelAdapter maps (in_channels -> ckpt_chan_n) with a 1x1 Conv (built from detected channels).
    - Time resize to 1024 by pad/trim.
    - Forward: encoder -> mean pool over N & embed_num -> Linear -> 1 logit
    """
    def __init__(self, cfg: Config, detected_in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = int(detected_in_channels)

        # Build the adapter to cfg.eegpt_channels using detected channels
        if self.in_channels != cfg.eegpt_channels:
            self.adapter = nn.Conv1d(self.in_channels, cfg.eegpt_channels, kernel_size=1, bias=False)
        else:
            self.adapter = nn.Identity()

        self.encoder = EEGTransformerEncoder(
            img_size=(cfg.eegpt_channels, cfg.eegpt_seq_len),
            patch_size=cfg.patch_size,
            patch_stride=cfg.patch_stride,
            embed_dim=cfg.embed_dim,
            embed_num=cfg.embed_num,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=4.0,
        )

        self.head = nn.Linear(cfg.embed_dim, 1)
        self.chan_ids = torch.arange(cfg.eegpt_channels).unsqueeze(0)  # (1, Ckpt)

    def _time_resize(self, x: torch.Tensor) -> torch.Tensor:
        T_in = x.shape[-1]
        T_tar = self.cfg.eegpt_seq_len
        if T_in == T_tar: return x
        if T_in > T_tar:  return x[:, :, :T_tar]
        return torch.nn.functional.pad(x, (0, T_tar - T_in))

    def forward(self, x):
        x = self.adapter(x)               # -> (B, Ckpt, T_in)
        x = self._time_resize(x)          # -> (B, Ckpt, 1024)
        feats = self.encoder(x, self.chan_ids.to(x.device))  # (B, N, E, D)
        feats = feats.mean(dim=1).mean(dim=1)                # (B, D)
        logit = self.head(feats)                             # (B, 1)
        return logit

# ======= CHECKPOINT LOADING (PyTorch 2.6 safe + chan_embed resize) =======
def _safe_load_ckpt(ckpt_path: str):
    """
    Robust loader for PyTorch 2.6:
    1) Try weights_only=False (OK if you TRUST the checkpoint).
    2) If TypeError (older torch), retry without the arg.
    3) If UnpicklingError, allowlist needed globals and retry with weights_only=True.
    """
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
    Falls back conservatively if keys are missing.
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
        # fallback: infer embed_dim from patch_embed.proj
        pe = pick("encoder.patch_embed.proj.weight",
                  "model.encoder.patch_embed.proj.weight")
        embed_dim = int(pe.shape[0]) if pe is not None else 512
        embed_num = 1

    return chan_n, embed_dim, embed_num

def load_eegpt_encoder_from_ckpt(model: nn.Module, ckpt_path: str):
    """
    Loads only the EEGPT encoder weights from a Lightning .ckpt into model.encoder (DP-safe).
    Also **resizes chan_embed** to match ckpt if needed (common cause of shape mismatch).
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

    target_encoder = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder

    # --- Handle chan_embed shape mismatch (e.g., ckpt has 62, model built with 128) ---
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

    # Load remaining state (strict=False to allow benign diffs)
    missing, unexpected = target_encoder.load_state_dict(enc_state, strict=False)
    print("[EEGPT load] Missing keys:", missing)
    print("[EEGPT load] Unexpected keys:", unexpected)

# ======= TRAIN / EVAL =======
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        y_prob.append(prob)
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
    return {"acc": acc, "bacc": bacc}

def main():
    cfg = CFG
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = setup_logging(cfg.output_dir, base_name="train_log")

    # Device selection inside code
    device = torch.device(f"cuda:{PRIMARY_LOCAL_INDEX}" if torch.cuda.is_available() else "cpu")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    print("Using device:", device)
    print("Log file:", log_path)

    # Data
    train_loader, val_loader, test_loader = make_loaders(cfg)

    # --- Auto-detect input shape from data (fixes 16/30-ch input) ---
    detected_C, detected_T = peek_input_shape(train_loader)
    if cfg.in_channels_hint is not None and cfg.in_channels_hint != detected_C:
        print(f"[warn] in_channels_hint={cfg.in_channels_hint} differs from detected {detected_C}; using detected.")
    if cfg.seq_len_in_hint is not None and cfg.seq_len_in_hint != detected_T:
        print(f"[warn] seq_len_in_hint={cfg.seq_len_in_hint} differs from detected {detected_T}; using detected.")
    print(f"[shape] detected input: C={detected_C}, T={detected_T}")

    # --- Inspect EEGPT ckpt to discover its true hyperparams (channels, width, summary tokens) ---
    assert os.path.isfile(cfg.ckpt_path), f"Checkpoint not found: {cfg.ckpt_path}"
    ckpt_chan_n, ckpt_embed_dim, ckpt_embed_num = inspect_eegpt_ckpt_meta(cfg.ckpt_path)
    print(f"[ckpt] encoder meta -> chan_n={ckpt_chan_n}, embed_dim={ckpt_embed_dim}, embed_num={ckpt_embed_num}")

    # --- Rebuild config to MATCH the ckpt (avoid size mismatches) ---
    cfg.eegpt_channels = ckpt_chan_n
    cfg.embed_dim      = ckpt_embed_dim
    cfg.embed_num      = ckpt_embed_num

    # Build model with detected input channels and ckpt-aligned encoder widths
    base_model = EEGPTStressClassifier(cfg, detected_in_channels=detected_C).to(device)

    # Optional multi-GPU (uses all visible GPUs listed above)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Wrapping model with nn.DataParallel over all visible GPUs...")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    # Load EEGPT encoder weights from Lightning .ckpt (PyTorch 2.6-safe + chan_embed resize)
    load_eegpt_encoder_from_ckpt(model, cfg.ckpt_path)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # PyTorch 2.6 AMP API
    scaler = torch.amp.GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=(cfg.use_amp and device.type == "cuda"))

    best_val = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                     enabled=(cfg.use_amp and device.type == "cuda")):
                logits = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1

        dur = time.time() - t0
        train_loss = epoch_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | loss={train_loss:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} | val_bacc={val_metrics['bacc']:.4f} | {dur:.1f}s")

        # Save
        state_to_save = (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
        to_save = {
            "epoch": epoch,
            "model": state_to_save,
            "optim": optimizer.state_dict(),
            "cfg": cfg.__dict__,
            "best_val": best_val,
        }
        if cfg.save_best_only:
            score = val_metrics["bacc"]
            if score > best_val:
                best_val = score
                to_save["best_val"] = best_val
                torch.save(to_save, os.path.join(cfg.output_dir, "checkpoint-best.pth"))
                print(f"  -> Saved BEST (bacc={best_val:.4f})")
        else:
            torch.save(to_save, os.path.join(cfg.output_dir, f"checkpoint-epoch{epoch+1:03d}.pth"))

    # Test with best
    best_path = os.path.join(cfg.output_dir, "checkpoint-best.pth") if cfg.save_best_only else None
    if best_path and os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded best: {best_path}")

    test_metrics = evaluate(model, test_loader, device)
    print("========== TEST ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print("==========================")

    # Export fine-tuned model weights only (nice for deployment)
    export_path = os.path.join(cfg.output_dir, "eegpt_stress_finetuned.pth")
    torch.save((model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()), export_path)
    print(f"Exported fine-tuned weights to: {export_path}")

if __name__ == "__main__":
    main()
