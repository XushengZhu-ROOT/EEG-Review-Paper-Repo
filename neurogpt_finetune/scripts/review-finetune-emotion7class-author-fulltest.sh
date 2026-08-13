#!/bin/bash
set -e
dataset=emotion7class
dataset_path="../seed_data"
matrix_p_path="../tMatrix_22x62_seed.npy"
gpu_id=0
pos_weight=-1.0  # 6分类任务（移除neutral），不需要pos_weight

# 数据预处理参数
num_chunks=2  # 4秒数据分成4个chunks
chunk_len=500  # 1秒 = 250个时间点 @ 250Hz（与预训练模型匹配）
chunk_ovlp=0   # 不重叠

# 模型参数
cls_head_layer='3ly'

# 作者config（全微调）
batch_size=32
lr=0.0001
wd=0.1

exp_name="author_exp_full_3ly_lr${lr}_wd${wd}_bs${batch_size}"

echo "--- 🏃 执行实验: ${exp_name} ---"
echo "LR: ${lr}, WD: ${wd}, Batch Size: ${batch_size}"

CUDA_VISIBLE_DEVICES=${gpu_id} python3 ../src/print_config.py \
--training-style='decoding' \
--num-decoding-classes=6 \
--training-steps=10000  \
--eval_every_n_steps=500 \
--log-every-n-steps=500 \
--num_chunks=${num_chunks} \
--per-device-training-batch-size=${batch_size} \
--per-device-validation-batch-size=${batch_size} \
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
--embedding-dim=1024  \
--pretrained-model='../pytorch_model.bin' \
--dataset-name=${dataset} \
--dst-data-path=${dataset_path} \
--matrix_p_path=${matrix_p_path} \
--seed=0 \
--pos_weight=${pos_weight} \
--log-dir="../results/${dataset}" \
--metric_for_best_model="eval_validation_bacc" \
--cls_head_layer=${cls_head_layer}

echo "--- 实验完成 ---"

