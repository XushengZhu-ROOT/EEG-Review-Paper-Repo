#!/bin/bash
# [KaggleERN bestval] 用已经训练好的最佳 checkpoint 做一次干净推理，不重新训练。
# lr/weight_decay/batch_size/classifier/freeze_type 会自动从 checkpoint_path 的目录名/
# 文件名解析出来（cbramod 自己的训练脚本就是用这个命名约定存的），不用手填。
#
# hpo_exp16（lr=1e-4, wd=2e-5, bs=512）是 结果统计/kaggle.csv 里 CBraMod 一行
# best val bacc 最高的一组（73.24%），真实 checkpoint 路径已经跟同事确认过。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

checkpoint_path="/work/HHRI-AI/YW/Yirong/CBraMod/models_weights/KaggleERN-posWeight0.413/hpo_exp16_lr0.0001_wd0.00002_bs512-all_patch_reps-all/best_model_epoch19_testAcc0.71324_testBacc0.68295.pth"
data_root="./cbramod_kaggleern_data"

python infer_kaggleern_checkpoint.py \
    --checkpoint_path "${checkpoint_path}" \
    --data_root "${data_root}" \
    --fold_results_dir ./fold_results_cbramod_kaggleern

python compute_metrics_from_npz.py --npz_dir ./fold_results_cbramod_kaggleern --task kaggleern --model cbramod --n_classes 2 --ci --n_bootstrap 1000
cat ./fold_results_cbramod_kaggleern/kaggleern_cbramod.json
