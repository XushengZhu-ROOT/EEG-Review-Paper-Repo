#!/bin/bash
# [Sleep] 用已经训练好的最佳 checkpoint 做一次干净推理，不重新训练。
# lr/weight_decay/batch_size/best_epoch 这些会自动从 checkpoint_path 的目录名/
# 文件名解析出来（cbramod 自己的训练脚本就是用这个命名约定存的），不用手填。
#
# 如果 checkpoint 挪了地方，改下面这一行就行，其它都不用动。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

checkpoint_path="/work/HHRI-AI/YW/Yirong/CBraMod/models_weights/Sleep/hpo_exp4_lr0.0001_wd0.00002_bs64-all_patch_reps-all/best_model_epoch28_valBacc0.77046_testBacc0.77713.pth"
data_root="./cbramod_sleep_data"

python infer_sleep_checkpoint.py \
    --checkpoint_path "${checkpoint_path}" \
    --data_root "${data_root}" \
    --fold_results_dir ./fold_results_cbramod_sleep

python compute_metrics_from_npz.py --npz_dir ./fold_results_cbramod_sleep --task sleep --model cbramod --n_classes 5 --ci --n_bootstrap 1000
cat ./fold_results_cbramod_sleep/sleep_cbramod.json
