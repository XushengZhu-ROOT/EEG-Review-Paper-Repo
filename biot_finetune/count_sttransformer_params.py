#!/usr/bin/env python3
"""
统计 STTransformer-review-finetune-SEED.sh 运行时的真实参数量
- 使用与 sh 完全相同的配置构建模型（run_multiclass_supervised.py 的 STTransformer 配置）
- 由程序遍历模型各子模块计算，非手算
- 输出类似同事表格的格式，不区分 Trainable/Non-trainable
"""

import sys
import os

# 确保能导入 model
import importlib.util

base = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "st_transformer", os.path.join(base, "model", "st_transformer.py")
)
st_transformer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st_transformer)
STTransformer = st_transformer.STTransformer

import torch


def count_params(module):
    """统计模块的总参数量（程序遍历）"""
    return sum(p.numel() for p in module.parameters(recurse=True))


def count_buffers(module):
    """统计模块的 buffer 数量"""
    return sum(b.numel() for b in module.buffers(recurse=True))


def main():
    # ========== 与 STTransformer-review-finetune-SEED.sh 完全一致的配置 ==========
    # 来自 run_multiclass_supervised.py 的 STTransformer 分支
    dataset = "SEED-ST"
    in_channels = 62
    n_classes = 6
    sampling_rate = 250
    sample_length = 4

    channel_legnth = int(sampling_rate * sample_length)  # 1000
    emb_size = 256
    depth = 4  # run_multiclass_supervised 中写死的 4 层

    print("=" * 70)
    print("2. STTransformer (SEED-ST 全微調)")
    print("   STTransformer-review-finetune-SEED.sh 真实参数量 - 由程序 output 计算")
    print("=" * 70)
    print(f"配置: dataset={dataset}, in_channels={in_channels}, n_classes={n_classes}")
    print(f"      sampling_rate={sampling_rate}, sample_length={sample_length}s")
    print(f"      channel_legnth={channel_legnth}, emb_size={emb_size}, depth={depth}")
    print()

    # 构建与 run_multiclass_supervised 完全相同的模型
    model = STTransformer(
        emb_size=emb_size,
        depth=depth,
        n_classes=n_classes,
        channel_legnth=channel_legnth,
        n_channels=in_channels,
    )

    # 按组件统计（程序遍历，非手算）
    chan_block = model.channel_attension
    patch_emb = model.patch_embedding
    transformer = model.transformer
    classification = model.classification

    p_chan = count_params(chan_block)
    p_patch = count_params(patch_emb)
    p_trans = count_params(transformer)
    p_cls = count_params(classification)

    buf_patch = count_buffers(patch_emb)
    buf_total = count_buffers(model)

    backbone_subtotal = p_chan + p_patch + p_trans
    total = backbone_subtotal + p_cls

    # 表格：Component | Params（不区分 Trainable/Non-trainable）
    rows = [
        ("ChannelAttention block (LayerNorm + CA + Dropout, ResidualAdd)", p_chan),
        ("PatchSTEmbedding (2×Conv1d + BatchNorm + Rearrange)", p_patch),
        (f"TransformerEncoder ({depth} layers)", p_trans),
        ("Backbone subtotal", backbone_subtotal),
        ("ClassificationHead (ELU + Linear 256→6)", p_cls),
        ("Total", total),
    ]

    # 可选：Transformer 每一层的参数量
    trans_per_layer = count_params(transformer) // depth if depth > 0 else 0
    print("Transformer 各层参数量（每层相同）:")
    for i in range(depth):
        layer = transformer[i]
        layer_params = count_params(layer)
        print(f"  Layer {i}: {layer_params:,}")
    print()

    def _fmt(n):
        return f"{n:,}"

    max_name = max(len(r[0]) for r in rows)
    print("参数表格（Component | Params）:")
    print(f"  {'Component':<{max_name}}  {'Params':>14}")
    print("  " + "-" * (max_name + 18))
    for name, n in rows:
        print(f"  {name:<{max_name}}  {_fmt(n):>14}")
    print()
    print(f"  Buffers: {_fmt(buf_total)} (BatchNorm running stats)")
    print(f"  Grand total: {_fmt(total + buf_total)}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
