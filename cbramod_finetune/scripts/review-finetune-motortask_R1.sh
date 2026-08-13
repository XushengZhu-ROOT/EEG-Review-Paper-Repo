#!/bin/bash
# =============================================================================
# MotorTask fine-tune launcher (R1: 支持受试者独立划分)
# =============================================================================
# 用法：
#   bash review-finetune-motortask_R1.sh
#
# 切换划分方式：改下面的 split_mode
#   - random_epoch         : 旧方法，读磁盘 train/val/test 随机 epoch 划分
#   - subject_independent  : R1 新方法，按受试者 18/1/1 严格划分（防泄漏）
# =============================================================================

dataset=MotorTask
dataset_path=./AllSubjects_Epochs
# 实验名可带 split 标记，方便和旧结果对比
exp_name=exp_author_config_R1
gpu_id=0
chan_size=20
window_size=1
epochs=50
lr=0.0005
wd=0.002
bs=64
classifier=all_patch_reps
seed=0

freeze_type=all

# -----------------------------------------------------------------------------
# [划分策略开关] 改这一处即可在新旧方法间切换
# -----------------------------------------------------------------------------
# 旧方法（随机 epoch 划分，存在相邻窗口泄漏风险；保留以便对比复现）：
#   split_mode=random_epoch
# R1 新方法（受试者独立，18 train / 1 val / 1 test）：
#   split_mode=subject_independent
split_mode=subject_independent

# [R1] single_fold_debug=True：只跑第一折 dry run，不做完整 LOSO
# 第一折默认：排序后 subjects[:-2] 训练，subjects[-2] 验证，subjects[-1] 测试
# （当前数据无 Sub12，排序末尾为 Sub20 / Sub21）
single_fold_debug=True

# [R1 可选] 手动指定 val/test 受试者；留空则用默认第一折（Sub20 / Sub21）
# 若要固定某折，取消注释并填写，例如：
#   val_subject=Sub20
#   test_subject=Sub21
val_subject=""
test_subject=""

# -----------------------------------------------------------------------------
# [保留] 旧实验对照示例：若要换回随机 epoch 划分，把上面改成：
#   split_mode=random_epoch
#   exp_name=exp_author_config
# 并可不传 --split_mode / --single_fold_debug（默认就是 random_epoch）
# -----------------------------------------------------------------------------

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--frozen"
else
    FREEZE_ARG=""
fi

# 可选受试者参数（仅 subject_independent 且手动指定时追加）
SUBJECT_ARGS=""
if [ -n "${val_subject}" ] && [ -n "${test_subject}" ]; then
    SUBJECT_ARGS="--val_subject ${val_subject} --test_subject ${test_subject}"
fi

# 模型保存目录带上 split_mode，避免覆盖旧的 random_epoch 权重
model_dir=./models_weights/${dataset}/${exp_name}-${classifier}-${freeze_type}-${split_mode}

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
    --classifier ${classifier} \
    --use_pretrained_weights True \
    --foundation_dir ./pretrained_weights/pretrained_weights.pth \
    --cuda ${gpu_id} \
    --model_dir ${model_dir} \
    --batch_size ${bs} \
    --lr ${lr} \
    --weight_decay ${wd} \
    --split_mode ${split_mode} \
    --single_fold_debug ${single_fold_debug} \
    ${SUBJECT_ARGS}
