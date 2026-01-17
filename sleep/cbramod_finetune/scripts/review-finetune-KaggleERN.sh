#!/bin/bash
set -e
cd /work/HHRI-AI/YW/Yirong/CBraMod/


#--------- KaggleERN - fintune all (全微調) ---------#
#---------           - author config (一組) -------------#
#---------           - 3 層 Classifier -------------#
dataset=KaggleERN
dataset_path=/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n56-cbramod
exp_name=exp_author_config
gpu_id=1
chan_size=56
window_size=3
epochs=50
lr=0.0001
wd=0.00002
bs=64
pos_weight=0.413
classifier=all_patch_reps # 三層
seed=0

freeze_type=all

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--frozen"
else
    FREEZE_ARG=""
fi

# python finetune_main.py \
#     --seed ${seed} \
#     --epochs ${epochs} \
#     --downstream_dataset ${dataset} \
#     --datasets_dir ${dataset_path} \
#     --pos_weight ${pos_weight} \
#     --optimizer  AdamW \
#     --num_of_classes 2 \
#     --channel_size ${chan_size} \
#     --window_size ${window_size} \
#     ${FREEZE_ARG} \
#     --classifier ${classifier} \
#     --use_pretrained_weights True \
#     --foundation_dir ./pretrained_weights/pretrained_weights.pth \
#     --cuda ${gpu_id} \
#     --model_dir /work/HHRI-AI/YW/Yirong/CBraMod/models_weights/${dataset}-posWeight${pos_weight}/${exp_name}-${classifier}-${freeze_type} \
#     --batch_size ${bs} \
#     --lr ${lr} \
#     --weight_decay ${wd}



#--------- KaggleERN - fintune all (全微調) ---------#
#---------           - hpo -------------#
#---------           - 3 層 Classifier -------------#
BS_LIST=(64 256 512)
LR_LIST=(0.0005 0.0001 0.00005)
WD_LIST=(0.002 0.00002)

exp_count=1
for bs in "${BS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for wd in "${WD_LIST[@]}"; do
        exp_name=hpo_exp${exp_count}_lr${lr}_wd${wd}_bs${bs}
        echo "--- [STARTING EXPERIMENT] ---"
        echo "BatchSize: ${bs}, LearningRate: ${lr}, WeightDecay: ${wd}"
        echo "Exp Name: ${exp_name}"
        echo "-------------------------------"
        python finetune_main.py \
            --seed ${seed} \
            --epochs ${epochs} \
            --downstream_dataset ${dataset} \
            --datasets_dir ${dataset_path} \
            --pos_weight ${pos_weight} \
            --optimizer  AdamW \
            --num_of_classes 2 \
            --channel_size ${chan_size} \
            --window_size ${window_size} \
            ${FREEZE_ARG} \
            --classifier ${classifier} \
            --use_pretrained_weights True \
            --foundation_dir ./pretrained_weights/pretrained_weights.pth \
            --cuda ${gpu_id} \
            --model_dir /work/HHRI-AI/YW/Yirong/CBraMod/models_weights/${dataset}-posWeight${pos_weight}/${exp_name}-${classifier}-${freeze_type} \
            --batch_size ${bs} \
            --lr ${lr} \
            --weight_decay ${wd}
        echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
        echo "" # 空一行方便閱讀

        # (4.4) ⭐ 這是新加的：計數器 + 1
        exp_count=$((exp_count + 1))
    done
  done
done



#--------- KaggleERN - linear probe (只訓練頭) ---------#
#---------           - author config (一組) -------------#
#---------           - 3 層 Classifier -------------#
dataset=KaggleERN
dataset_path=/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n56-cbramod
exp_name=exp_author_config
gpu_id=1
chan_size=56
window_size=3
epochs=50
lr=0.0001
wd=0.00002
bs=64
pos_weight=0.413
classifier=all_patch_reps # 三層
seed=0

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
    --pos_weight ${pos_weight} \
    --optimizer  AdamW \
    --num_of_classes 2 \
    --channel_size ${chan_size} \
    --window_size ${window_size} \
    ${FREEZE_ARG} \
    --classifier ${classifier} \
    --use_pretrained_weights True \
    --foundation_dir ./pretrained_weights/pretrained_weights.pth \
    --cuda ${gpu_id} \
    --model_dir /work/HHRI-AI/YW/Yirong/CBraMod/models_weights/${dataset}-posWeight${pos_weight}/${exp_name}-${classifier}-${freeze_type} \
    --batch_size ${bs} \
    --lr ${lr} \
    --weight_decay ${wd}