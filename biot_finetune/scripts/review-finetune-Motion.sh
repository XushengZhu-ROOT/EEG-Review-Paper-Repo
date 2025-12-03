#!/bin/bash
set -e

cd ~/EEG-Review-Paper-Repo/biot_finetune/

dataset=Motion
gpu_id=0

dataset_channels=16
sample_length=3
epochs=50
pretrain_model_channels=16
pretrain_path=pretrained-models/EEG-PREST-${pretrain_model_channels}-channels.ckpt

classifier_type=CBraMod_3lyStyle_LayerNorm-BIOT
layers=3ly

lr=0.001
wd=0.00001
bs=256
pos_weight=0.413

freeze_type=all

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi

output_dir=${dataset}-posWeight${pos_weight}-author
exp_name=exp_author_config-${classifier_type}_${layers}-${freeze_type}

echo "===================================================="
echo "Running Motion Author Config: ${exp_name}"
echo "===================================================="

CUDA_VISIBLE_DEVICES=${gpu_id} python run_multiclass_supervised.py \
    --exp_name ${exp_name} \
    --dataset ${dataset} \
    --n_classes 6 \
    --dataset_channels ${dataset_channels} \
    --in_channels ${pretrain_model_channels} \
    --sampling_rate 200 \
    --token_size 200 \
    --hop_length 100 \
    --sample_length ${sample_length} \
    --batch_size ${bs} \
    --lr ${lr} \
    --weight_decay ${wd} \
    --pos_weight ${pos_weight} \
    --epochs ${epochs} \
    --model ${classifier_type} \
    --pretrain_model_path ${pretrain_path} \
    --output_dir ${output_dir} \
    ${FREEZE_ARG}


BS_LIST=(256 512)
LR_LIST=(0.005 0.001 0.0005)
WD_LIST=(0.001 0.00001)

count=0

echo "===================================================="
echo "Starting Motion HPO sweep..."
echo "===================================================="

for bs in "${BS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for wd in "${WD_LIST[@]}"; do
        count=$((count + 1))
        exp_name=exp_hpo${count}-${classifier_type}_${layers}-${freeze_type}

        echo "================================================================"
        echo "Running HPO experiment: ${exp_name}"
        echo "Batch Size (bs): ${bs}"
        echo "Learning Rate (lr): ${lr}"
        echo "Weight Decay (wd): ${wd}"
        echo "================================================================"

        CUDA_VISIBLE_DEVICES=${gpu_id} python run_multiclass_supervised.py \
            --exp_name ${exp_name} \
            --dataset ${dataset} \
            --n_classes 6 \
            --dataset_channels ${dataset_channels} \
            --in_channels ${pretrain_model_channels} \
            --sampling_rate 200 \
            --token_size 200 \
            --hop_length 100 \
            --sample_length ${sample_length} \
            --batch_size ${bs} \
            --lr ${lr} \
            --weight_decay ${wd} \
            --pos_weight ${pos_weight} \
            --epochs ${epochs} \
            --model ${classifier_type} \
            --pretrain_model_path ${pretrain_path} \
            --output_dir ${dataset}-HPO \
            ${FREEZE_ARG}

        echo "--- Finished HPO experiment: ${exp_name} ---"
        echo ""
    done
  done
done