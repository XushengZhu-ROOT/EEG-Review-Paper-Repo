#!/usr/bin/env python
"""
统计各 review-fietune-*.sh 运行时的可训练参数量。

用法:
  python count_params.py [--config CONFIG] [--linear]
  --config: motor_author | stress_swien (默认 motor_author)
  --linear: freeze_type=linear_probe，否则 freeze_type=all
"""
import sys
from argparse import Namespace

# 注册模型
import modeling_finetune  # noqa: F401

from run_class_finetuning import get_models

# 解析命令行
argv = sys.argv[1:]
use_linear = '--linear' in argv
config_name = 'motor_author'
if '--config' in argv:
    i = argv.index('--config')
    if i + 1 < len(argv):
        config_name = argv[i + 1]

# 配置表: 与各 shell 脚本一致
CONFIGS = {
    'motor_author': {
        'name': 'review-fietune-Motor_author.sh',
        'model': 'labram_base_patch200_200_cbramod3lyclassifier',
        'nb_classes': 6,
        'channel_size': 20,
        'classifier_window_size': 1,
    },
    'stress_swien': {
        'name': 'review-fietune-stress_swien.sh',
        'model': 'labram_base_patch200_200',  # 1ly
        'nb_classes': 1,
        'channel_size': 30,
        'classifier_window_size': 5,
    },
    'kaggleern': {
        'name': 'review-fietune-KaggleERN.sh',
        'model': 'labram_base_patch200_200_cbramod3lyclassifier',  # 3ly
        'nb_classes': 1,
        'channel_size': 55,
        'classifier_window_size': 3,
    },
    'seed_author': {
        'name': 'review-finetune-Seed_authorLinear.sh',
        'model': 'labram_base_patch200_200_cbramod3lyclassifier',  # 3ly
        'nb_classes': 6,
        'channel_size': 62,
        'classifier_window_size': 1,
    },
}

cfg = CONFIGS.get(config_name, CONFIGS['motor_author'])
args = Namespace(
    model=cfg['model'],
    nb_classes=cfg['nb_classes'],
    drop=0.0,
    drop_path=0.1,
    attn_drop_rate=0.0,
    use_mean_pooling=True,
    init_scale=0.001,
    rel_pos_bias=False,
    abs_pos_emb=True,
    layer_scale_init_value=0.1,
    qkv_bias=False,
    channel_size=cfg['channel_size'],
    classifier_window_size=cfg['classifier_window_size'],
    classifier_dropout=0.1,
    freeze_backbone=use_linear,
)

model = get_models(args)

# 若 freeze_backbone，与 run_class_finetuning 一致地冻结
if args.freeze_backbone:
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, 'fc_norm'):
        for param in model.fc_norm.parameters():
            param.requires_grad = True
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True

def _param_count(m):
    if m is None:
        return 0
    if hasattr(m, 'numel'):  # Parameter/Tensor
        return m.numel()
    return sum(p.numel() for p in m.parameters())

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total - trainable

# Backbone 分解，便于与同事文档对照
parts = {
    'patch_embed': _param_count(model.patch_embed),
    'cls+pos+time_embed': _param_count(model.cls_token)
        + _param_count(model.pos_embed)
        + _param_count(model.time_embed),
    'blocks': sum(_param_count(b) for b in model.blocks),
    'fc_norm': _param_count(model.fc_norm) if model.fc_norm else 0,
    'head': _param_count(model.head),
}
backbone = parts['patch_embed'] + parts['cls+pos+time_embed'] + parts['blocks'] + parts['fc_norm']

mode = "linear_probe (仅 head)" if args.freeze_backbone else "all (全参数微调)"
print("=" * 50)
print(f"{cfg['name']} | {mode}")
print("=" * 50)
print(f"总参数量:     {total:,}")
print(f"可训练参数量: {trainable:,}")
print(f"冻结参数量:   {frozen:,}")
print("-" * 50)
print("Backbone 分解 (与同事文档对照):")
print(f"  patch_embed:     {parts['patch_embed']:>10,}  (同事: 576)")
print(f"  cls+pos+time:    {parts['cls+pos+time_embed']:>10,}  (同事: 32,400)")
print(f"  blocks(12层):    {parts['blocks']:>10,}  (同事: 5,789,760)")
print(f"  fc_norm:         {parts['fc_norm']:>10,}  (同事: 400)")
print(f"  backbone小计:    {backbone:>10,}  (同事: 5,823,136)")
print(f"  head:            {parts['head']:>10,}  (同事 6类: 1,206)")
print("=" * 50)
