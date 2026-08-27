#!/bin/bash
# [KaggleERN bestval] 用结果统计/kaggle.csv 里 Labram 一行 best val bacc 最高的
# 超参数组合重跑一遍（lr=9e-4, wd=5e-2, bs=512, val_bacc=68.91%, best_epoch=20，
# all/全微调, 3层 classifier "swien config"）。训练时用 val_bacc 最好的那个 epoch
# 存 kaggleern_labram_val.npz/_test.npz/kaggleern_labram.json。
set -e
cd /work/HHRI-AI/YW/Yirong/LaBramFinetune/

dataset=KaggleERN
dataset_path=/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/EEGPT_Data/KaggleERN/s42_n55-labram
gpu_id=0
chan_size=55
epochs=50
seed=0
pos_weight=0.413
lr=9e-4
wd=5e-2
bs=512
freeze_type=all
model_type=3ly
exp_name=bestval_lr${lr}_wd${wd}_bs${bs}

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
     --output_dir ./new_ckpt/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
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
     --disable_qkv_bias \
     --task_name kaggleern \
     --fold_results_dir ./fold_results_labram_kaggleern

python compute_metrics_from_npz.py --npz_dir ./fold_results_labram_kaggleern --task kaggleern --model labram --n_classes 2 --ci --n_bootstrap 1000
cat ./fold_results_labram_kaggleern/kaggleern_labram.json
