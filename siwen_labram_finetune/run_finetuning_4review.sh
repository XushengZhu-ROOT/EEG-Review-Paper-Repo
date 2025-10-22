#!/bin/bash

# Stress Dataset setting
chan=30
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
     --output_dir ./checkpoints/finetune_stress_noleak_${chan}chan_no400up_seed_siwen42-4review-default \
     --log_dir ./log/finetune_stress_noleak_${chan}chan_no400up_seed_siwen42-4review-default \
     --dataset_path augmented_data/Stress_noleak_${chan}chan_no400up_seed_siwen42 \
     --channel_size ${chan} \
     --finetune ./checkpoints/labram-base.pth \
     --model labram_base_patch200_200 \
     --batch_size 512 \
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
     --dataset Stress \
     --disable_qkv_bias \
     --seed 0