#!/bin/bash
# [KaggleERN bestval] 用结果统计/kaggle.csv 里 NeuroLM 一行 best val bacc 最高的
# 超参数组合重跑一遍（lr=5e-5, wd=1e-2, eeg_bs=48, text_bs=12, val_bacc=63.65%,
# best_epoch=4）。跑完会在 --results_dir 下存 kaggleern_neurolm_val.npz/_test.npz/
# kaggleern_neurolm.json。
#
# 按几张卡改 n_gpu/gpu_id 即可，这里给的是单卡默认值。
set -e
n_gpu=1
gpu_id=0
# 固定參數
dataset_dir=/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n55-neurolm
test_data_dir=/work/HHRI-AI/YW/Yirong/NeuroLM/data/text
NeuroLM_path=checkpoints/NeuroLM-B.pt

dataset=KaggleERN
chan_size=55
exp_name=bestval_lr5e-05_wd0.01_textbs12_instbs48
out_dir=checkpoints/${dataset}/${exp_name}

epochs=5
min_lr=5e-5
adamw_b1=0.9
adamw_b2=0.95

lr=5e-5
wd=0.01
tbs=12
ibs=48

CUDA_VISIBLE_DEVICES=${gpu_id} MP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=${n_gpu} train_instruction.py \
    --dataset_dir ${dataset_dir} \
    --text_data_dir ${test_data_dir} \
    --out_dir ${out_dir} \
    --NeuroLM_path ${NeuroLM_path} \
    --chan_size ${chan_size} \
    --eeg_batch_size ${ibs} \
    --text_batch_size ${tbs} \
    --epochs ${epochs} \
    --learning_rate ${lr} \
    --min_lr ${min_lr} \
    --beta1 ${adamw_b1} \
    --beta2 ${adamw_b2} \
    --weight_decay ${wd} \
    --save_ckpt \
    --model_name neurolm \
    --task_name kaggleern \
    --results_dir ./fold_results_neurolm_kaggleern

python compute_metrics_from_npz.py --npz_dir ./fold_results_neurolm_kaggleern --task kaggleern --model neurolm --n_classes 2 --ci --n_bootstrap 1000
cat ./fold_results_neurolm_kaggleern/kaggleern_neurolm.json

echo "========================================"
echo "Best-val rerun completed."
echo "========================================"
