#!/bin/bash

dataset=SEED-Emotion
dataset_path=./seed_data
exp_name=exp_author_config
gpu_id=0
chan_size=62
window_size=4
epochs=50
lr=0.0001
wd=0.00002
bs=64
seed=62

freeze_type=linear_probe

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--frozen"
else
    FREEZE_ARG=""
fi

python finetune_main.py \
    --seed ${seed} \
    --epochs ${epochs} \
    --downstream_dataset ${dataset} \
    --datasets_dir ${dataset_path} \
    --optimizer  AdamW \
    --num_of_classes 6 \
    --channel_size ${chan_size} \
    --window_size ${window_size} \
    ${FREEZE_ARG} \
    --use_pretrained_weights \
    --foundation_dir CSBrain.pth \
    --cuda ${gpu_id} \
    --model_dir ./models_weights/${dataset}/${exp_name}-${freeze_type} \
    --batch_size ${bs} \
    --lr ${lr} \
    --weight_decay ${wd} \
    --dropout 0.1 \
    --label_smoothing 0.1
