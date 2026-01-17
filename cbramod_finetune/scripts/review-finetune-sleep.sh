#!/bin/bash
set -e
cd /home/dung/Documents/EEG-Review-Paper-Repo/cbramod_finetune

dataset=ISRUC
dataset_path=/home/dung/Documents/EEG-Review-Paper-Repo/data/isruc_cbramod
exp_name=exp_author_config
gpu_id=0
chan_size=6
window_size=30
epochs=50
num_classes=5
classifier=all_patch_reps # 3 Labram_style_classifier # 1層 
seed=0
freeze_type=all # linear_probe

#--------- STRESS - Linear Probe (只訓練頭) ---------#
#---------           - author config (一組) -------------#
#---------           - 1 層 Classifier -------------#

# lr=0.0001
# wd=0.00002
# bs=64

# # 1. 只訓練頭
# freeze_type=linear_probe

# if [ "${freeze_type}" = "linear_probe" ]; then
#     FREEZE_ARG="--frozen"
# else
#     # 如果不是，我們就傳入一個空字串，等於什麼都不加
#     FREEZE_ARG=""
# fi

# python finetune_main.py \
#     --seed ${seed} \
#     --epochs ${epochs} \
#     --downstream_dataset ${dataset} \
#     --datasets_dir ${dataset_path} \
#     --optimizer  AdamW \
#     --num_of_classes ${num_classes} \
#     --channel_size ${chan_size} \
#     --window_size ${window_size} \
#     ${FREEZE_ARG} \
#     --classifier ${classifier} \
#     --use_pretrained_weights True \
#     --foundation_dir ./pretrained_weights/pretrained_weights.pth \
#     --cuda ${gpu_id} \
#     --model_dir checkpoints/${dataset}/${exp_name}-${classifier}-${freeze_type} \
#     --batch_size ${bs} \
#     --lr ${lr} \
#     --weight_decay ${wd}


#--------- STRESS - fintune all (全微調) ---------#
#---------           - author config (一組) -------------#
#---------           - 1 層 Classifier -------------#
# 2. 微調全部
# freeze_type=all

# if [ "${freeze_type}" = "linear_probe" ]; then
#     FREEZE_ARG="--frozen"
# else
#     FREEZE_ARG=""
# fi

# python finetune_main.py \
#     --seed ${seed} \
#     --epochs ${epochs} \
#     --downstream_dataset ${dataset} \
#     --datasets_dir ${dataset_path} \
#     --optimizer  AdamW \
#     --num_of_classes ${num_classes} \
#     --channel_size ${chan_size} \
#     --window_size ${window_size} \
#     ${FREEZE_ARG} \
#     --classifier ${classifier} \
#     --use_pretrained_weights True \
#     --foundation_dir ./pretrained_weights/pretrained_weights.pth \
#     --cuda ${gpu_id} \
#     --model_dir checkpoints/${dataset}/${exp_name}-${classifier}-${freeze_type} \
#     --batch_size ${bs} \
#     --lr ${lr} \
#     --weight_decay ${wd}



# #--------- STRESS - Linear Probe (只訓練頭) ---------#
# #---------           - hpo -------------#
# #---------           - 1 層 Classifier -------------#

# # Batch size (bs): [256, 512, 1024]
# # Learning rate (lr): [1e-3, 5e-4, 5e-5]
# # Weight decay (wd): [0.0, 1e-3]
# BS_LIST=(64 256 512)
# LR_LIST=(0.0005 0.0001 0.00005)
# WD_LIST=(0.002 0.00002)

# # 只訓練頭
# freeze_type=linear_probe
# if [ "${freeze_type}" = "linear_probe" ]; then
#     FREEZE_ARG="--frozen"
# else
#     FREEZE_ARG=""
# fi


# exp_count=1
# for bs in "${BS_LIST[@]}"; do
#     for lr in "${LR_LIST[@]}"; do
#         for wd in "${WD_LIST[@]}"; do
#             exp_name=hpo_exp${exp_count}_lr${lr}_wd${wd}_bs${bs}
#             echo "--- [STARTING EXPERIMENT] ---"
#             echo "BatchSize: ${bs}, LearningRate: ${lr}, WeightDecay: ${wd}"
#             echo "Exp Name: ${exp_name}"
#             echo "-------------------------------"
#             python finetune_main.py \
#                 --seed ${seed} \
#                 --epochs ${epochs} \
#                 --downstream_dataset ${dataset} \
#                 --datasets_dir ${dataset_path} \
#                 --optimizer  AdamW \
#                 --num_of_classes ${num_classes} \
#                 --channel_size ${chan_size} \
#                 --window_size ${window_size} \
#                 ${FREEZE_ARG} \
#                 --classifier ${classifier} \
#                 --use_pretrained_weights True \
#                 --foundation_dir ./pretrained_weights/pretrained_weights.pth \
#                 --cuda ${gpu_id} \
#                 --model_dir checkpoints/${dataset}/${exp_name}-${classifier}-${freeze_type} \
#                 --batch_size ${bs} \
#                 --lr ${lr} \
#                 --weight_decay ${wd}
#             echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
#             echo "" 

#             exp_count=$((exp_count + 1))
#         done
#     done
# done


# #--------- STRESS - finetune (全微調) ---------#
# #---------           - hpo -------------#
# #---------           - 1 層 Classifier -------------#

BS_LIST=(64 256 512)
LR_LIST=(0.0005 0.0001 0.00005)
WD_LIST=(0.002 0.00002)
# 微調全部
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--frozen"
else
    FREEZE_ARG=""
fi


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
                --optimizer  AdamW \
                --num_of_classes ${num_classes} \
                --channel_size ${chan_size} \
                --window_size ${window_size} \
                ${FREEZE_ARG} \
                --classifier ${classifier} \
                --use_pretrained_weights True \
                --foundation_dir ./pretrained_weights/pretrained_weights.pth \
                --cuda ${gpu_id} \
                --model_dir checkpoints/${dataset}/${exp_name}-${classifier}-${freeze_type} \
                --batch_size ${bs} \
                --lr ${lr} \
                --weight_decay ${wd}
            echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
            echo "" 

            exp_count=$((exp_count + 1))
        done
    done
done