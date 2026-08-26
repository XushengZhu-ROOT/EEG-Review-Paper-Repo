#!/bin/bash
# neurogpt Sleep 最佳超参数正式跑（不做 HPO 网格搜索，只跑这一组）。
# 跑完会在 --fold-results-dir 下存 sleep_neurogpt_val.npz / _test.npz / sleep_neurogpt.json，
# 供事后算 per-stage recall / confusion matrix / macro-F1 / kappa / 95% CI。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=sleep5class
dataset_path="../sleep_data"
matrix_p_path="../tMatrix_22x6_seed.npy"
gpu_id=0
pos_weight=-1.0  # 5分类任务，不需要pos_weight

num_chunks=30
chunk_len=250
chunk_ovlp=0

cls_head_layer='3ly'

lr=3e-4
wd=0.1
bs=128

MAX_STEPS=10000
exp_name="bestval_lr${lr}_wd${wd}_bs${bs}"

CUDA_VISIBLE_DEVICES=${gpu_id} python3 ../src/train_gpt.py \
    --training-style='decoding' \
    --num-decoding-classes=5 \
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
    --pretrained-model='../pytorch_model.bin' \
    --dataset-name=${dataset} \
    --dst-data-path=${dataset_path} \
    --matrix_p_path=${matrix_p_path} \
    --seed=0 \
    --pos_weight=${pos_weight} \
    --log-dir="../results/${dataset}" \
    --metric_for_best_model="eval_validation_bacc" \
    --cls_head_layer=${cls_head_layer} \
    --task-name=sleep \
    --model-name=neurogpt \
    --fold-results-dir="../fold_results_neurogpt_sleep"

cd ..
python3 compute_metrics_from_npz.py --npz_dir ./fold_results_neurogpt_sleep --task sleep --model neurogpt --n_classes 5 --ci --n_bootstrap 1000
cat ./fold_results_neurogpt_sleep/sleep_neurogpt.json
