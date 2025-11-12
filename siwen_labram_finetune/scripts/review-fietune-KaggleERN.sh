# #!/bin/bash
set -e
# conda activate labram
cd /work/HHRI-AI/YW/Yirong/LaBramFinetune/

#--------- KaggleERN - fintune all (全微調) ---------#
#---------           - swien config (一組) -------------#
#---------           - 3 層 Classifier -------------#
### 1. 微調全部, 3層, swien config

dataset=KaggleERN
dataset_path=/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n55-labram
gpu_id=2
exp_name=exp_swien_config
chan_size=55
epochs=50
seed=0
pos_weight=0.413
lr=5e-4
wd=0.05
bs=512
freeze_type=all
model_type=3ly
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
     --output_dir ./checkpoints/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
     --log_dir ./log/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
     --dataset_path ${dataset_path} \
     --channel_size ${chan_size} \
     --pos_weight ${pos_weight} \
     ${FREEZE_ARG} \
     --finetune ./checkpoints/labram-base.pth \
     --model ${MODEL_ARG} \
      --classifier_window_size 3 \
     --abs_pos_emb \
     --dist_eval \
     --dataset ${dataset} \
     --seed ${seed} \
     --epochs ${epochs} \
     --lr ${lr} \
     --weight_decay ${wd} \
     --batch_size ${bs} \
     --min_lr 1e-6 \
     --layer_decay 0.65 \
     --drop_path 0.1 \
     --opt_betas 0.9 0.999 \
     --update_freq 1 \
     --warmup_epochs 3 \
     --save_ckpt_freq 5 \
     --disable_rel_pos_bias \
     --disable_qkv_bias


#--------- KaggleERN - fintune all (全微調) ---------#
#---------           - hpo -------------#
#---------           - 3 層 Classifier -------------#
### 2. 微調全部, 3層, swien config + 超參數搜尋
# 範圍
BS_LIST=(256 512 1024)
LR_LIST=(0.0009 0.0005 0.00009)
WD_LIST=(0.05 0.005 0.0001)

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
            --output_dir ./checkpoints/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
            --log_dir ./log/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
            --dataset_path ${dataset_path} \
            --channel_size ${chan_size} \
            --pos_weight ${pos_weight} \
            ${FREEZE_ARG} \
            --finetune ./checkpoints/labram-base.pth \
            --model ${MODEL_ARG} \
            --classifier_window_size 3 \
            --abs_pos_emb \
            --dist_eval \
            --dataset ${dataset} \
            --seed ${seed} \
            --epochs ${epochs} \
            --lr ${lr} \
            --weight_decay ${wd} \
            --batch_size ${bs} \
            --min_lr 1e-6 \
            --layer_decay 0.65 \
            --drop_path 0.1 \
            --opt_betas 0.9 0.999 \
            --update_freq 1 \
            --warmup_epochs 3 \
            --save_ckpt_freq 5 \
            --disable_rel_pos_bias \
            --disable_qkv_bias
            echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
            echo "" 

            exp_count=$((exp_count + 1))
    done
  done
done



#--------- KaggleERN - linear probe (只訓練頭) ---------#
#---------           - swien config (一組) -------------#
#---------           - 3 層 Classifier -------------#
### 3. 只訓練頭, 3層, swien config

dataset=KaggleERN
dataset_path=/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n55-labram
gpu_id=2
exp_name=exp_swien_config
chan_size=55
epochs=50
seed=0
pos_weight=0.413
lr=5e-4
wd=0.05
bs=512
freeze_type=linear_probe
model_type=3ly
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
     --output_dir ./checkpoints/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
     --log_dir ./log/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
     --dataset_path ${dataset_path} \
     --channel_size ${chan_size} \
     --pos_weight ${pos_weight} \
     ${FREEZE_ARG} \
     --finetune ./checkpoints/labram-base.pth \
     --model ${MODEL_ARG} \
      --classifier_window_size 3 \
     --abs_pos_emb \
     --dist_eval \
     --dataset ${dataset} \
     --seed ${seed} \
     --epochs ${epochs} \
     --lr ${lr} \
     --weight_decay ${wd} \
     --batch_size ${bs} \
     --min_lr 1e-6 \
     --layer_decay 0.65 \
     --drop_path 0.1 \
     --opt_betas 0.9 0.999 \
     --update_freq 1 \
     --warmup_epochs 3 \
     --save_ckpt_freq 5 \
     --disable_rel_pos_bias \
     --disable_qkv_bias