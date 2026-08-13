#!/usr/bin/env python3
"""
统计 review-finetune-SEED.sh linear_probe 时的可训练参数量
- 使用与 sh 完全相同的配置构建模型
- 应用 freeze_backbone 后的真实参数量
- 输出 backbone / linear 头 的可训练参数
- 输出每一块的参数量（由程序遍历模型/checkpoint 计算，非手算）
"""

import os
import torch
import torch.nn as nn


def count_params(module, trainable_only=False):
    """统计模块的参数量"""
    total = 0
    for p in module.parameters(recurse=True):
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def run_with_full_model(dataset, dataset_channels, pretrain_model_channels, pretrain_path, n_classes, classifier_type, freeze_type):
    """使用完整模型统计（需 linear_attention_transformer）"""
    from model import CBraMod_3lyStyle_LayerNorm_Ada_BIOT

    model = CBraMod_3lyStyle_LayerNorm_Ada_BIOT(
        input_chan_size=dataset_channels,
        n_classes=n_classes,
        n_channels=pretrain_model_channels,
        n_fft=200,
        hop_length=100,
    )

    try:
        ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=True)
        model.biot.load_state_dict(ckpt, strict=True)
        print(f"[OK] 已加载预训练: {pretrain_path}")
    except Exception as e:
        print(f"[警告] 预训练加载失败: {e}")

    if freeze_type == "linear_probe":
        for p in model.biot.parameters():
            p.requires_grad = False
        print("[OK] 已冻结 model.biot (backbone)")

    return model


def run_fallback_from_ckpt(dataset_channels, pretrain_model_channels, pretrain_path, n_classes):
    """
    无法导入完整模型时：从 checkpoint 统计 backbone，用 nn 构建 head 统计
    所有数字均由程序计算/遍历得出
    """
    # 1. Backbone: 从 checkpoint 遍历计算
    ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=True)
    backbone_total = sum(p.numel() for p in ckpt.values())

    # 2. 按 prefix 聚合 backbone 各块（程序遍历 keys）
    from collections import defaultdict
    prefix_sums = defaultdict(int)
    for k, v in ckpt.items():
        # transformer.layers.layers.0.xxx -> block 0
        prefix_sums[k] = v.numel()

    # 按模块分组（遍历 checkpoint keys）
    patch_emb = sum(prefix_sums[k] for k in ckpt if k.startswith("patch_embedding."))
    pos_enc = sum(prefix_sums[k] for k in ckpt if "positional" in k)
    ch_tokens = sum(prefix_sums[k] for k in ckpt if k.startswith("channel_tokens.") or k == "index")
    trans_total = sum(prefix_sums[k] for k in ckpt if k.startswith("transformer."))

    # transformer 各 block
    block_params = defaultdict(int)
    for k in ckpt:
        if "transformer.layers.layers." in k:
            parts = k.split(".")
            # transformer.layers.layers.X.xxx
            idx = parts.index("layers")
            if idx + 2 < len(parts):
                block_idx = int(parts[idx + 2])
                block_params[block_idx] += ckpt[k].numel()

    # 3. Head: 构建模块统计（与 biot.py 一致）
    emb_size = 256
    chan_conv = nn.Conv1d(dataset_channels, pretrain_model_channels, 1)
    fc_norm = nn.LayerNorm(emb_size)
    cls_head = nn.Sequential(
        nn.Linear(emb_size, emb_size),
        nn.Linear(emb_size, emb_size),
        nn.Linear(emb_size, n_classes),
    )

    head_total = (
        sum(p.numel() for p in chan_conv.parameters()) +
        sum(p.numel() for p in fc_norm.parameters()) +
        sum(p.numel() for p in cls_head.parameters())
    )

    return {
        "model": None,
        "backbone_total": backbone_total,
        "backbone_trainable": 0,  # linear_probe 下 backbone 冻结
        "head_total": head_total,
        "head_trainable": head_total,
        "total_params": backbone_total + head_total,
        "total_trainable": head_total,
        "fallback": True,
        "block_params": dict(block_params),
        "patch_emb": patch_emb,
        "pos_enc": pos_enc,
        "ch_tokens": ch_tokens,
        "trans_total": trans_total,
        "chan_conv": sum(p.numel() for p in chan_conv.parameters()),
        "fc_norm": sum(p.numel() for p in fc_norm.parameters()),
        "classifier": sum(p.numel() for p in cls_head.parameters()),
        "cls_layers": [
            sum(p.numel() for p in m.parameters()) for m in cls_head if hasattr(m, "parameters")
        ],
    }


def _fmt(n):
    return f"{n:,}"


def collect_table_data(result, model=None):
    """收集表格数据，返回 (rows, buffers, grand_total)。格式与同事一致。"""
    rows = []
    buffers_total = 0

    if result.get("fallback"):
        index_count = 16
        ch_tokens_only = result["ch_tokens"] - index_count  # Embedding 16×256=4096, index=16
        bb_components = [
            ("PatchFrequencyEmbedding (Linear 101→256)", result["patch_emb"], 0),
            ("Channel Tokens (Embedding 16×256)", ch_tokens_only, 0),
            ("Positional Encoding (sinusoidal buffer)", 0, 0),
            ("LinearAttention Transformer (4 layers)", result["trans_total"], 0),
            ("index (device-tracking vector)", 0, index_count),
        ]
        bb_t = sum(r[1] for r in bb_components)
        bb_n = sum(r[2] for r in bb_components)
        rows = [("chan_conv (Conv1d 62→16)", result["chan_conv"], 0)]
        rows.extend(bb_components)
        rows.append(("Backbone subtotal", bb_t, bb_n))
        rows.append(("fc_norm (LayerNorm 256)", result["fc_norm"], 0))
        rows.append(("ClassificationHead_3ly (3×Linear 256→256→6)", result["classifier"], 0))
        total_train = result["chan_conv"] + result["fc_norm"] + result["classifier"]
        total_non = index_count
        rows.append(("Total", total_train, total_non))
        buffers_total = result.get("pos_enc", 256000)
        grand = total_train + total_non + buffers_total
        return rows, buffers_total, grand

    # Full model：从 model 遍历得到真实数值
    biot = model.biot

    def _params(m):
        if isinstance(m, nn.Parameter):
            return (m.numel(), 0) if m.requires_grad else (0, m.numel())
        params = list(m.parameters(recurse=True))
        t = sum(p.numel() for p in params if p.requires_grad)
        n = sum(p.numel() for p in params if not p.requires_grad)
        return t, n

    rows = []
    if hasattr(model, "chan_conv"):
        cc_t, cc_n = _params(model.chan_conv)
        rows.append(("chan_conv (Conv1d 62→16)", cc_t, cc_n))

    patch_t, patch_n = _params(biot.patch_embedding)
    rows.append(("PatchFrequencyEmbedding (Linear 101→256)", patch_t, patch_n))

    ch_t, ch_n = _params(biot.channel_tokens)
    rows.append(("Channel Tokens (Embedding 16×256)", ch_t, ch_n))

    rows.append(("Positional Encoding (sinusoidal buffer)", 0, 0))

    trans_t, trans_n = _params(biot.transformer)
    rows.append(("LinearAttention Transformer (4 layers)", trans_t, trans_n))

    idx_t, idx_n = _params(biot.index)
    rows.append(("index (device-tracking vector)", idx_t, idx_n))

    bb_t = patch_t + ch_t + trans_t + idx_t
    bb_n = patch_n + ch_n + trans_n + idx_n
    rows.append(("Backbone subtotal", bb_t, bb_n))

    if hasattr(model, "fc_norm"):
        fn_t, fn_n = _params(model.fc_norm)
        rows.append(("fc_norm (LayerNorm 256)", fn_t, fn_n))

    if hasattr(model, "classifier"):
        cl_t, cl_n = _params(model.classifier)
        rows.append(("ClassificationHead_3ly (3×Linear 256→256→6)", cl_t, cl_n))

    total_t = result["total_trainable"]
    total_n = result["total_params"] - result["total_trainable"]
    rows.append(("Total", total_t, total_n))

    for name, buf in biot.named_buffers():
        buffers_total += buf.numel()

    grand = result["total_params"] + buffers_total
    return rows, buffers_total, grand


def print_table(rows, buffers_total, grand_total):
    """按同事的表格格式输出"""
    max_name = max(len(r[0]) for r in rows)
    print(f"  {'Component':<{max_name}}  {'Trainable':>14}  {'Non-trainable':>14}")
    print("  " + "-" * (max_name + 32))
    for name, tr, nt in rows:
        print(f"  {name:<{max_name}}  {_fmt(tr):>14}  {_fmt(nt):>14}")
    print()
    print(f"  Buffers: {_fmt(buffers_total)} (Positional Encoding)")
    print(f"  Grand total: {_fmt(grand_total)}")


def main():
    # ========== 与 review-finetune-SEED.sh 完全一致的配置 ==========
    dataset = "SEED"
    dataset_channels = 62
    pretrain_model_channels = 16
    pretrain_path = "pretrained-models/EEG-PREST-16-channels.ckpt"
    n_classes = 6
    classifier_type = "CBraMod_3lyStyle_LayerNorm-BIOT"
    freeze_type = "linear_probe"

    use_fallback = False
    try:
        from model import CBraMod_3lyStyle_LayerNorm_Ada_BIOT
    except ImportError as e:
        print(f"[提示] 无法导入完整模型 ({e})，使用 checkpoint + head 构建的 fallback 模式")
        use_fallback = True

    if use_fallback:
        if not os.path.exists(pretrain_path):
            print(f"错误: 预训练文件不存在 {pretrain_path}")
            return
        result = run_fallback_from_ckpt(
            dataset_channels, pretrain_model_channels, pretrain_path, n_classes
        )
    else:
        model = run_with_full_model(
            dataset, dataset_channels, pretrain_model_channels, pretrain_path, n_classes, classifier_type, freeze_type
        )
        backbone_total = count_params(model.biot, trainable_only=False)
        backbone_trainable = count_params(model.biot, trainable_only=True)
        head_total = 0
        head_trainable = 0
        for name, m in [("chan_conv", model.chan_conv), ("fc_norm", model.fc_norm), ("classifier", model.classifier)]:
            if hasattr(model, name):
                h = getattr(model, name)
                head_total += count_params(h, trainable_only=False)
                head_trainable += count_params(h, trainable_only=True)
        result = {
            "model": model,
            "backbone_total": backbone_total,
            "backbone_trainable": backbone_trainable,
            "head_total": head_total,
            "head_trainable": head_trainable,
            "total_params": backbone_total + head_total,
            "total_trainable": head_trainable,
            "fallback": False,
        }

    # ========== 输出结果（按同事表格格式，数值由程序运行得到）==========
    print("\n" + "=" * 70)
    print("1. CBraMod_3lyStyle_LayerNorm-BIOT (SEED linear_probe)")
    print("   review-finetune-SEED.sh 真实参数量 - 由 count_trainable_params.py 输出")
    print("=" * 70)
    print(f"配置: dataset={dataset}, channels={dataset_channels}, n_classes={n_classes}, freeze={freeze_type}")
    if result.get("fallback"):
        print("(Fallback 模式: 从 checkpoint 遍历 + head 模块构建统计)")
    print()

    model_for_table = result.get("model")
    rows, buffers_total, grand_total = collect_table_data(result, model_for_table)
    print("参数表格（Component | Trainable | Non-trainable）:")
    print()
    print_table(rows, buffers_total, grand_total)

    print()
    print("=" * 70)
    print(f"结论: linear_probe 时仅 chan_conv + fc_norm + ClassificationHead_3ly 可训练")
    print(f"      可训练参数量 = {result['total_trainable']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
