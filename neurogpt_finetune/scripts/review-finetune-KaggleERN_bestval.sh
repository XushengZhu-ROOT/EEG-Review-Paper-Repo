#!/bin/bash
# [KaggleERN bestval] 用结果统计/kaggle.csv 里 NeuroGPT 一行 best val bacc 最高的
# 超参数组合重跑一遍（lr=6e-5, wd=1e-2, bs=16, val_bacc=62.12%, best_step=2000,
# all/全微调）。跑完会在 --fold-results-dir 下存 kaggleern_neurogpt_val.npz/_test.npz/
# kaggleern_neurogpt.json。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=KaggleERN
dataset_path="/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n56-neurogpt"
gpu_id=0
sample_rate=250
pos_weight=0.413

num_chunks=2
window_size=2
chunk_len=$((window_size * sample_rate))
chunk_ovlp=250

lr=6e-5
wd=1e-2
bs=16

MAX_STEPS=10000
exp_name="bestval_lr${lr}_wd${wd}_bs${bs}"

CUDA_VISIBLE_DEVICES=${gpu_id} python3 ../src/train_gpt.py \
    --training-style='decoding' \
    --num-decoding-classes=2 \
    --training-steps=${MAX_STEPS} \
    --eval_every_n_steps=500 \
    --log-every-n-steps=500 \
    --num_chunks=${num_chunks} \
    --per-device-training-batch-size=${bs} \
    --per-device-validation-batch-size=${bs} \
    --chunk_len=${chunk_len} \
    --chunk_ovlp=${chunk_ovlp} \
    --run-name=${exp_name} \
    --ft-only-encoder='True' \
    --fold_i=0 \
    --num-encoder-layers=6 \
    --num-hidden-layers=6 \
    --learning-rate=${lr} \
    --weight-decay=${wd} \
    --use-encoder='True' \
    --embedding-dim=1024 \
    --pretrained-model='../pretrained_model/pytorch_model.bin' \
    --dataset-name=${dataset} \
    --dst-data-path=${dataset_path} \
    --seed=0 \
    --pos_weight=${pos_weight} \
    --log-dir="../results/${dataset}" \
    --metric_for_best_model="eval_validation_bacc" \
    --task-name=kaggleern \
    --model-name=neurogpt \
    --fold-results-dir="../fold_results_neurogpt_kaggleern"

cd ..
python3 compute_metrics_from_npz.py --npz_dir ./fold_results_neurogpt_kaggleern --task kaggleern --model neurogpt --n_classes 2 --ci --n_bootstrap 1000
cat ./fold_results_neurogpt_kaggleern/kaggleern_neurogpt.json
