#!/bin/bash
# 可视化 R1 受试者独立实验结果（无 GUI，图存到 ./viz_R1）
# 用法：bash run_visualize_motor_R1.sh

# R1（默认）
EXP_DIR=./models_weights/MotorTask/exp_author_config_R1-all_patch_reps-all-subject_independent
SAVE_DIR=./viz_R1

# 若要对比旧的随机 epoch 结果，改成下面两行：
# EXP_DIR=./models_weights/MotorTask/exp_author_config-all_patch_reps-all
# SAVE_DIR=./viz_random_epoch

python log_visualize_motor_R1.py \
    --exp_dir "${EXP_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --no_show
