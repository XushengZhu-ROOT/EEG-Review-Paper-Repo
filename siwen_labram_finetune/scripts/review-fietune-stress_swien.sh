#!/bin/bash
set -e
# conda activate labram
cd /work/HHRI-AI/YW/Yirong/LaBramFinetune/

#--------- STRESS - Linear Probe (只訓練頭) ---------#
#---------           - swien config (一組) -------------#
#---------           - 1 層 Classifier -------------#
# 1. 只訓練頭, 1層

dataset=Stress
dataset_path=./augmented_data/Stress_noleak_30chan_no400up_seed_siwen42
gpu_id=2
chan_size=30
model_type=1ly
seed=0

epochs=50
lr=1e-3 # 5e-4
wd=0.05 # 0.05

exp_name=exp_swien_config-lr${lr}-wd${wd}
freeze_type=linear_probe
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi
if [ "${model_type}" = "3ly" ]; then
    MODEL_ARG="labram_base_patch200_200_cbramod3lyclassifier"
else
    MODEL_ARG="labram_base_patch200_200"
fi
CUDA_VISIBLE_DEVICES=${gpu_id} python run_class_finetuning.py \
     --output_dir ./checkpoints/${dataset}_search/${exp_name}-${model_type}-${freeze_type} \
     --log_dir ./log/${dataset}_search/${exp_name}-${model_type}-${freeze_type} \
     --dataset_path ${dataset_path} \
     --channel_size ${chan_size} \
     ${FREEZE_ARG} \
     --finetune ./checkpoints/labram-base.pth \
     --model ${MODEL_ARG} \
      --classifier_window_size 5 \
     --dataset ${dataset} \
     --seed ${seed} \
     --lr ${lr} \
     --min_lr 1e-6 \
     --opt_betas 0.9 0.999 \
     --weight_decay ${wd} \
     --update_freq 1 \
     --warmup_epochs 3 \
     --epochs ${epochs} \
     --layer_decay 0.65 \
     --drop_path 0.1 \
     --dist_eval \
     --save_ckpt_freq 5 \
     --disable_rel_pos_bias \
     --abs_pos_emb \
     --disable_qkv_bias 


#--------- STRESS - finetune all (全微調) ---------#
#---------           - swien config (一組) -------------#
#---------           - 1 層 Classifier -------------#
# 2. 微調全部

dataset=Stress
dataset_path=./augmented_data/Stress_noleak_30chan_no400up_seed_siwen42
gpu_id=1
exp_name=exp_swien_config
chan_size=30
model_type=1ly
seed=0

freeze_type=all
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi
if [ "${model_type}" = "3ly" ]; then
    MODEL_ARG="labram_base_patch200_200_cbramod3lyclassifier"
else
    MODEL_ARG="labram_base_patch200_200"
fi
freeze_type=all
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi
CUDA_VISIBLE_DEVICES=${gpu_id} python run_class_finetuning.py \
     --output_dir ./checkpoints/${dataset}_search-noPosWeight/${exp_name}-${model_type}-${freeze_type} \
     --log_dir ./log/${dataset}_search-noPosWeight/${exp_name}-${model_type}-${freeze_type} \
     --dataset_path ${dataset_path} \
     --channel_size ${chan_size} \
     ${FREEZE_ARG} \
     --finetune ./checkpoints/labram-base.pth \
     --model ${MODEL_ARG} \
      --classifier_window_size 5 \
     --dataset ${dataset} \
     --seed ${seed} \
     --lr 5e-4 \
     --min_lr 1e-6 \
     --opt_betas 0.9 0.999 \
     --weight_decay 0.05 \
     --update_freq 1 \
     --warmup_epochs 3 \
     --epochs 50 \
     --layer_decay 0.65 \
     --drop_path 0.1 \
     --dist_eval \
     --save_ckpt_freq 5 \
     --disable_rel_pos_bias \
     --abs_pos_emb \
     --disable_qkv_bias 



#--------- STRESS - Linear Probe (只訓練頭) ---------#
#---------           - hpo -------------#
#---------           - 1 層 Classifier -------------#
# 範圍
# Batch size (bs): [256, 512, 1024]
# Learning rate (lr): [1e-3, 5e-4, 5e-5]
# Weight decay (wd): [0.0, 1e-3]
BS_LIST=(256 512 1024)
LR_LIST=(0.005 0.001 0.0005)
WD_LIST=(0.001 0.0001)

freeze_type=linear_probe
model_type=1ly
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi
if [ "${model_type}" = "3ly" ]; then
    MODEL_ARG="labram_base_patch200_200_cbramod3lyclassifier"
else
    MODEL_ARG="labram_base_patch200_200"
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
     CUDA_VISIBLE_DEVICES=${gpu_id} python run_class_finetuning.py \
            --output_dir ./checkpoints/${dataset}_search-noPosWeight/${exp_name}-${model_type}-${freeze_type} \
            --log_dir ./log/${dataset}_search-noPosWeight/${exp_name}-${model_type}-${freeze_type} \
            --dataset_path ${dataset_path} \
            --channel_size ${chan_size} \
            ${FREEZE_ARG} \
            --finetune ./checkpoints/labram-base.pth \
            --model ${MODEL_ARG} \
            --classifier_window_size 5 \
            --batch_size ${bs} \
            --lr ${lr} \
            --weight_decay ${wd} \
            --dataset ${dataset} \
            --seed ${seed} \
            --min_lr 1e-6 \
            --opt_betas 0.9 0.999 \
            --update_freq 1 \
            --warmup_epochs 3 \
            --epochs 100 \
            --layer_decay 0.65 \
            --drop_path 0.1 \
            --dist_eval \
            --save_ckpt_freq 5 \
            --disable_rel_pos_bias \
            --abs_pos_emb \
            --disable_qkv_bias 
     echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
     echo "" 

     exp_count=$((exp_count + 1))
    done
  done
done

#--------- STRESS - finetune all (全微調) ---------#
#---------           - hpo -------------#
#---------           - 1 層 Classifier -------------#
# 微調全部
BS_LIST=(256 512 1024)
LR_LIST=(0.001 0.0005 0.0001)
WD_LIST=(0.01 0.05 0.001)

freeze_type=all
model_type=1ly
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi
if [ "${model_type}" = "3ly" ]; then
    MODEL_ARG="labram_base_patch200_200_cbramod3lyclassifier"
else
    MODEL_ARG="labram_base_patch200_200"
fi

exp_count=1
for bs in "${BS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for wd in "${WD_LIST[@]}"; do
     exp_name=swien_config_hpo_exp${exp_count}_lr${lr}_wd${wd}_bs${bs}
     echo "--- [STARTING EXPERIMENT] ---"
     echo "BatchSize: ${bs}, LearningRate: ${lr}, WeightDecay: ${wd}"
     echo "Exp Name: ${exp_name}"
     echo "-------------------------------"
     CUDA_VISIBLE_DEVICES=${gpu_id} python run_class_finetuning.py \
          --output_dir ./checkpoints/${dataset}_search-noPosWeight/${exp_name}-${model_type}-${freeze_type} \
          --log_dir ./log/${dataset}_search-noPosWeight/${exp_name}-${model_type}-${freeze_type} \
          --dataset_path ${dataset_path} \
          --channel_size ${chan_size} \
          ${FREEZE_ARG} \
          --finetune ./checkpoints/labram-base.pth \
          --model ${MODEL_ARG} \
          --classifier_window_size 5 \
          --batch_size ${bs} \
          --lr ${lr} \
          --weight_decay ${wd} \
          --abs_pos_emb \
          --dist_eval \
          --dataset ${dataset} \
          --seed ${seed} \
          --min_lr 1e-6 \
          --opt_betas 0.9 0.999 \
          --layer_decay 0.65 \
          --drop_path 0.1 \
          --disable_rel_pos_bias \
          --disable_qkv_bias 
     echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
     echo "" 

     exp_count=$((exp_count + 1))
    done
  done
done
