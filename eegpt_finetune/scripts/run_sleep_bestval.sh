#!/bin/bash
# =============================================================================
# EEGPT Sleep fine-tune launcher -- single best-val hyperparameter run
# =============================================================================
# linear_probe_EEGPT_Sleep.py 本来只有一个网格搜索入口（BS_LIST x LR_LIST x
# WD_LIST），没有单独跑一组超参数的 CLI。这里用的是新加的 --single_run 开关
# （不传的话，脚本行为完全不变，还是原来的网格搜索），指向真实的 sleep_data，
# 跑满 max_epochs。跑完会在 output/${TASK_NAME}_bs..._lr..._wd..._freeze0/ 下
# 存 sleep_eegpt_val.npz / sleep_eegpt_test.npz / sleep_eegpt.json（best_state
# 已经在训练过程中按 val_balanced_accuracy 追踪，不需要 --save_ckpt 落盘）。
#
# Usage:
#   bash scripts/run_sleep_bestval.sh
# =============================================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python linear_probe_EEGPT_Sleep.py \
    --single_run \
    --data_root sleep_data \
    --task_name EEGPT_Sleep \
    --batch_size 64 \
    --max_lr 1e-3 \
    --weight_decay 1e-2 \
    --max_epochs 50

run_dir=$(ls -td output/EEGPT_Sleep_bs64_lr0.001_wd0.01_freeze0 2>/dev/null | head -1)
if [ -n "${run_dir}" ]; then
    python compute_metrics_from_npz.py --npz_dir "${run_dir}" --task sleep --model eegpt --n_classes 5 --ci --n_bootstrap 1000
    cat "${run_dir}/sleep_eegpt.json"
fi
