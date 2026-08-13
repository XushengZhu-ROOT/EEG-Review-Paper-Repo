#!/bin/bash
# 导出 cbramod 测试集预测结果，用于 McNemar 检验
# 输出: test_predictions.txt (epoch_id, pred, true, correct)

cd "$(dirname "$0")"

MODEL_DIR="models_weights/MotorTask/exp_author_config-all_patch_reps-all"
DATASETS_DIR="AllSubjects_Epochs"

python export_test_predictions.py \
    --model_dir "$MODEL_DIR" \
    --datasets_dir "$DATASETS_DIR" \
    --cuda 0
