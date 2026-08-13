#!/usr/bin/env python
"""
生成与同事文档格式一致的参数量报告。
所有数字来自真实模型 program output，非手算。

覆盖:
  - Stress (1ly): review-fietune-stress_swien.sh - 一层分类头, 二分类
  - KaggleERN (3ly): review-fietune-KaggleERN.sh - 三层分类头, 二分类
  - Seed (3ly): review-finetune-Seed_authorLinear.sh - 三层分类头, 6分类
  - Backbone 为共享部分

用法: python param_report.py [--config seed_author]  # 仅输出 seed_author 时使用
"""
import sys
from argparse import Namespace

import modeling_finetune  # noqa: F401
from run_class_finetuning import get_models


def _param_count(m):
    if m is None:
        return 0
    if hasattr(m, 'numel'):
        return m.numel()
    return sum(p.numel() for p in m.parameters())


def _get_parts(model):
    """从模型实际统计各块参数量"""
    return {
        'TemporalConv': _param_count(model.patch_embed),
        'cls_token': _param_count(model.cls_token),
        'pos_embed': _param_count(model.pos_embed),
        'time_embed': _param_count(model.time_embed),
        'blocks': sum(_param_count(b) for b in model.blocks),
        'fc_norm': _param_count(model.fc_norm) if model.fc_norm else 0,
        'head': _param_count(model.head),
    }

def _trainable_count(m):
    if m is None:
        return 0
    if hasattr(m, "requires_grad") and hasattr(m, "numel"):
        return m.numel() if m.requires_grad else 0
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _apply_freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, 'fc_norm') and model.fc_norm:
        for p in model.fc_norm.parameters():
            p.requires_grad = True
    if hasattr(model, 'head'):
        for p in model.head.parameters():
            p.requires_grad = True


def run_config(name, model, nb_classes, channel_size, window_size, freeze, is_3ly=False):
    args = Namespace(
        model=model,
        nb_classes=nb_classes,
        drop=0.0, drop_path=0.1, attn_drop_rate=0.0,
        use_mean_pooling=True, init_scale=0.001,
        rel_pos_bias=False, abs_pos_emb=True, layer_scale_init_value=0.1, qkv_bias=False,
        channel_size=channel_size,
        classifier_window_size=window_size,
        classifier_dropout=0.1,
        freeze_backbone=freeze,
    )
    m = get_models(args)
    if freeze:
        _apply_freeze(m)
    parts = _get_parts(m)
    parts_trainable = {
        'TemporalConv': _trainable_count(m.patch_embed),
        'cls_token': _trainable_count(m.cls_token),
        'pos_embed': _trainable_count(m.pos_embed),
        'time_embed': _trainable_count(m.time_embed),
        'blocks': sum(_trainable_count(b) for b in m.blocks),
        'fc_norm': _trainable_count(m.fc_norm) if m.fc_norm else 0,
        'head': _trainable_count(m.head),
    }
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    backbone = (parts['TemporalConv'] + parts['cls_token'] + parts['pos_embed'] +
                parts['time_embed'] + parts['blocks'] + parts['fc_norm'])
    cls_embed = parts['cls_token'] + parts['pos_embed'] + parts['time_embed']
    parts['cls_embed'] = cls_embed
    return {
        'name': name,
        'parts': parts,
        'parts_trainable': parts_trainable,
        'total': total,
        'trainable': trainable,
        'backbone': backbone,
        'cls_embed': cls_embed,
        'is_3ly': is_3ly,
    }


def print_table(r, mode_str):
    p = r['parts']
    print(f"\n{'='*60}")
    print(f"  {r['name']} | {mode_str}")
    print(f"{'='*60}")
    print(f"  {'Component':<40} {'Parameters':>12}  {'Non-trainable':>14}")
    print(f"  {'-'*40} {'-'*12}  {'-'*14}")
    print(f"  {'TemporalConv (3xConv2d + GroupNorm)':<40} {p['TemporalConv']:>12,}  {0:>14,}")
    print(f"  {'cls_token + pos_embed + time_embed':<40} {p['cls_embed']:>12,}  {0:>14,}")
    print(f"  {'Transformer blocks (12 layers)':<40} {p['blocks']:>12,}  {0:>14,}")
    print(f"  {'fc_norm (LayerNorm 200)':<40} {p['fc_norm']:>12,}  {0:>14,}")
    print(f"  {'Backbone subtotal':<40} {r['backbone']:>12,}  {0:>14,}")
    head_desc = r.get('head_desc', "3ly head (fc_layer2+fc_layer3)" if r['is_3ly'] else "1ly head (Linear 200→1)")
    print(f"  {f'Classification head ({head_desc})':<40} {p['head']:>12,}  {0:>14,}")
    print(f"  {'Total':<40} {r['total']:>12,}  {0:>14,}")
    print(f"{'='*60}")
    print(f"  Grand total: {r['total']:,}")
    if r.get('freeze'):
        print(f"  [Linear probe 实际可训练] fc_norm + head: {r['trainable']:,}")

def print_two_column_table(r_linear, r_all, title):
    tl = r_linear['parts_trainable']
    ta = r_all['parts_trainable']
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Component':<40} {'Linear':>12}  {'All':>12}")
    print(f"  {'-'*40} {'-'*12}  {'-'*12}")
    print(f"  {'Conv encoder (3×1D Conv + GN + GELU)':<40} {tl['TemporalConv']:>12,}  {ta['TemporalConv']:>12,}")
    print(f"  {'Spatial embedding (channel identity)':<40} {tl['pos_embed']:>12,}  {ta['pos_embed']:>12,}")
    print(f"  {'Temporal embedding (time index)':<40} {tl['time_embed']:>12,}  {ta['time_embed']:>12,}")
    print(f"  {'[CLS] token (global aggregator)':<40} {tl['cls_token']:>12,}  {ta['cls_token']:>12,}")
    print(f"  {'Transformer encoder (12 layers)':<40} {tl['blocks']:>12,}  {ta['blocks']:>12,}")
    print(f"  {'Final norm (LayerNorm 200)':<40} {tl['fc_norm']:>12,}  {ta['fc_norm']:>12,}")
    print(f"  {'Backbone subtotal':<40} {r_linear['backbone']:>12,}  {r_all['backbone']:>12,}")
    print(f"  {'Classification head':<40} {tl['head']:>12,}  {ta['head']:>12,}")
    print(f"  {'Total trainable':<40} {r_linear['trainable']:>12,}  {r_all['trainable']:>12,}")
    print(f"{'='*60}")


def main():
    argv = sys.argv[1:]
    only_config = None
    if '--config' in argv:
        i = argv.index('--config')
        if i + 1 < len(argv):
            only_config = argv[i + 1]

    print("\n" + "="*60)
    print("  LaBraM-Base 参数量报告 (程序实际输出)")
    print("="*60)

    if only_config == 'seed_author':
        # 仅输出 Seed
        r_linear = run_config(
            "review-finetune-Seed_authorLinear.sh (3层头, 6分类)",
            "labram_base_patch200_200_cbramod3lyclassifier",
            nb_classes=6, channel_size=62, window_size=1,
            freeze=True, is_3ly=True,
        )
        r_linear['freeze'] = True
        r_linear['head_desc'] = "3ly head (fc_layer2+fc_layer3, 200→6)"
        print_table(r_linear, "linear_probe (仅训练 fc_norm + head)")

        r_all = run_config(
            "review-finetune-Seed_authorLinear.sh (3层头)",
            "labram_base_patch200_200_cbramod3lyclassifier",
            nb_classes=6, channel_size=62, window_size=1,
            freeze=False, is_3ly=True,
        )
        r_all['head_desc'] = "3ly head (fc_layer2+fc_layer3, 200→6)"
        print_table(r_all, "all (全参数微调)")

        print_two_column_table(
            r_linear,
            r_all,
            "Seed: 线性与全量可训练参数对照 (按模块, 程序输出)",
        )

        print("\n" + "="*60)
        print("  Seed 汇总")
        print("="*60)
        print(f"  Backbone:                    {r_linear['backbone']:>12,}")
        print(f"  Linear probe 可训练:         {r_linear['trainable']:>12,}")
        print("="*60 + "\n")
        return

    # 1. Stress - 1ly, 二分类, linear probe
    r1 = run_config(
        "review-fietune-stress_swien.sh (1层头, 二分类)",
        "labram_base_patch200_200",
        nb_classes=1, channel_size=30, window_size=5,
        freeze=True, is_3ly=False,
    )
    r1['freeze'] = True
    print_table(r1, "linear_probe (仅训练 head)")

    # 2. Stress - 1ly, all
    r2 = run_config(
        "review-fietune-stress_swien.sh (1层头)",
        "labram_base_patch200_200",
        nb_classes=1, channel_size=30, window_size=5,
        freeze=False, is_3ly=False,
    )
    print_table(r2, "all (全参数微调)")

    # 3. KaggleERN - 3ly, 二分类, linear probe
    r3 = run_config(
        "review-fietune-KaggleERN.sh (3层头, 二分类)",
        "labram_base_patch200_200_cbramod3lyclassifier",
        nb_classes=1, channel_size=55, window_size=3,
        freeze=True, is_3ly=True,
    )
    r3['freeze'] = True
    print_table(r3, "linear_probe (仅训练 fc_norm + head)")

    # 4. KaggleERN - 3ly, all
    r4 = run_config(
        "review-fietune-KaggleERN.sh (3层头)",
        "labram_base_patch200_200_cbramod3lyclassifier",
        nb_classes=1, channel_size=55, window_size=3,
        freeze=False, is_3ly=True,
    )
    print_table(r4, "all (全参数微调)")

    # 汇总表
    print("\n" + "="*60)
    print("  汇总: Backbone 与 Linear Head 可训练参数量")
    print("="*60)
    print(f"  Backbone (共享):                    {r1['backbone']:>12,}")
    print(f"  1ly head 可训练 (fc_norm+Linear):   {r1['trainable']:>12,}  (Stress linear)")
    print(f"  3ly head 可训练 (fc_norm+3ly MLP):  {r3['trainable']:>12,}  (KaggleERN linear)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
