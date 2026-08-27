#!/bin/bash
# =============================================================================
# EEGPT KaggleERN fine-tune launcher -- single best-val hyperparameter run
# =============================================================================
# 结果统计/kaggle.csv 里 EEGPT 一行 best val bacc 最高的超参数组合
# （lr=4e-4, wd=1e-2, bs=64, val_bacc=55.74%, best_epoch=1, author config/全微调）。
# 用的是 linear_probe_EEGPT_KaggleERN.py 新加的 --single_run 开关（不传的话，脚本
# 行为完全不变，还是原来的网格搜索），跑完会在
# output/EEGPT_KaggleERN_bs64_lr0.0004_wd0.01_freeze0/ 下存
# kaggleern_eegpt_val.npz / kaggleern_eegpt_test.npz / kaggleern_eegpt.json。
#
# Usage:
#   bash scripts/run_kaggleern_bestval.sh
# =============================================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python linear_probe_EEGPT_KaggleERN.py \
    --single_run \
    --data_root kaggle_data \
    --task_name EEGPT_KaggleERN \
    --batch_size 64 \
    --max_lr 4e-4 \
    --weight_decay 1e-2 \
    --max_epochs 50

run_dir=$(ls -td output/EEGPT_KaggleERN_bs64_lr0.0004_wd0.01_freeze0 2>/dev/null | head -1)
if [ -n "${run_dir}" ]; then
    python compute_metrics_from_npz.py --npz_dir "${run_dir}" --task kaggleern --model eegpt --n_classes 2 --ci --n_bootstrap 1000
    cat "${run_dir}/kaggleern_eegpt.json"
fi
