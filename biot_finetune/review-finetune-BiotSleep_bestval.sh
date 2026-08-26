#!/bin/bash
set -e

dataset=BiotSleep
gpu_id=0

dataset_channels=6
sample_length=30
epochs=50
pretrain_model_channels=16
pretrain_path=pretrained-models/EEG-PREST-${pretrain_model_channels}-channels.ckpt

classifier_type=CBraMod_3lyStyle_LayerNorm-BIOT
layers=3ly

lr=1e-3
wd=0.001
bs=256

freeze_type=all

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi

exp_name=bestval-${classifier_type}_${layers}-${freeze_type}
output_dir=${dataset}-${classifier_type}

echo "===================================================="
echo "Running BiotSleep Best-Val Config: ${exp_name}"
echo "Dataset: ${dataset}"
echo "Channels: ${dataset_channels}"
echo "Classes: 5"
echo "Sample Length: ${sample_length}s (30s @ 200Hz = 6000 points)"
echo "lr=${lr} wd=${wd} bs=${bs}"
echo "===================================================="

CUDA_VISIBLE_DEVICES=${gpu_id} python run_multiclass_supervised.py \
    --exp_name ${exp_name} \
    --dataset ${dataset} \
    --n_classes 5 \
    --dataset_channels ${dataset_channels} \
    --in_channels ${pretrain_model_channels} \
    --sampling_rate 200 \
    --token_size 200 \
    --hop_length 100 \
    --sample_length ${sample_length} \
    --batch_size ${bs} \
    --lr ${lr} \
    --weight_decay ${wd} \
    --epochs ${epochs} \
    --model ${classifier_type} \
    --pretrain_model_path ${pretrain_path} \
    --output_dir ${output_dir} \
    --task_name sleep \
    --model_name biot \
    --fold_results_dir ./fold_results_biot_sleep \
    ${FREEZE_ARG}

python compute_metrics_from_npz.py --npz_dir ./fold_results_biot_sleep --task sleep --model biot --n_classes 5 --ci --n_bootstrap 1000
