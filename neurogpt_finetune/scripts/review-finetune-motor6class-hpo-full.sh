#!/bin/bash
set -e
dataset=motor6class
dataset_path="../AllSubjects_Epochs"
matrix_p_path="../tMatrix_22x20_motor.npy"
gpu_id=0
pos_weight=-1.0  # 6分类任务，不需要pos_weight

# 数据预处理参数
num_chunks=2
chunk_len=250  # 1秒 = 250个时间点 @ 250Hz（与预训练模型匹配）
chunk_ovlp=0   # 不重叠

# 模型参数
cls_head_layer='3ly'

# 超参数搜索范围（全微调）
LR_LIST=(3e-4 1e-4 6e-5)
WD_LIST=(0.01 0.1)
BS_LIST=(32 64 128)

# 训练参数（用于判断实验是否完成）
MAX_STEPS=10000

# 检查实验是否完成的函数
check_experiment_completed() {
    local exp_name=$1
    local exp_dir="../results/${dataset}/${exp_name}-0"
    echo "exp_dir: $exp_dir"
    
    # 检查实验目录是否存在
    if [ ! -d "$exp_dir" ]; then
        echo "exp_dir not found"
        return 1  # 目录不存在，实验未完成
    fi
    
    # 检查train_history.csv是否存在
    local train_history="${exp_dir}/train_history.csv"
    if [ ! -f "$train_history" ]; then
        echo "train_history.csv not found in $exp_dir"
        return 1  # 训练历史文件不存在，实验未完成
    fi
    
    # 检查训练是否达到max_steps（读取最后一行，提取step值）
    local last_line=$(tail -n 1 "$train_history" 2>/dev/null)
    local last_step=$(echo "$last_line" | cut -d',' -f1)
    
    # 检查step是否为数字且达到max_steps
    if ! [[ "$last_step" =~ ^[0-9]+$ ]] || [ "$last_step" -lt "$MAX_STEPS" ]; then
        echo "last_step: $last_step is not a number or less than $MAX_STEPS"
        return 1  # 训练未达到max_steps，实验未完成
    fi
    
    # 检查测试集预测文件是否存在（说明测试集预测已完成）
    local test_predictions="${exp_dir}/test_predictions.npy"
    local test_labels="${exp_dir}/test_label_ids.npy"
    if [ ! -f "$test_predictions" ] || [ ! -f "$test_labels" ]; then
        echo "test_predictions or test_labels not found in $exp_dir"
        return 1  # 测试集预测文件不存在，实验未完成
    fi
    
    # 所有检查都通过，实验已完成
    return 0
}

exp_count=1

for batch_size in "${BS_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
        for wd in "${WD_LIST[@]}"; do
            exp_name="hpo_exp${exp_count}_3ly_lr${lr}_wd${wd}_bs${batch_size}"
            
            # 检查实验是否已完成
            if check_experiment_completed "$exp_name"; then
                echo "--- ✅ 实验 #${exp_count}: ${exp_name} 已完成，跳过 ---"
                echo "LR: ${lr}, WD: ${wd}, Batch Size: ${batch_size}"
                echo ""
                exp_count=$((exp_count + 1))
                continue
            fi
            
            echo "--- 🏃 执行实验 #${exp_count}: ${exp_name} ---"
            echo "LR: ${lr}, WD: ${wd}, Batch Size: ${batch_size}"
            
            CUDA_VISIBLE_DEVICES=${gpu_id} python3 ../src/train_gpt.py \
            --training-style='decoding' \
            --num-decoding-classes=6 \
            --training-steps=${MAX_STEPS}  \
            --eval_every_n_steps=500 \
            --log-every-n-steps=500 \
            --num_chunks=${num_chunks} \
            --per-device-training-batch-size=${batch_size} \
            --per-device-validation-batch-size=${batch_size} \
            --chunk_len=${chunk_len} \
            --chunk_ovlp=${chunk_ovlp} \
            --run-name=${exp_name} \
            --ft-only-encoder='True' \
            --fold_i=0 \
            --num-encoder-layers=6 \
            --num-hidden-layers=6 \
            --learning-rate=${lr} \
            --weight-decay=${wd} \
            --use-encoder='True' \
            --embedding-dim=1024  \
            --pretrained-model='../pytorch_model.bin' \
            --dataset-name=${dataset} \
            --dst-data-path=${dataset_path} \
            --matrix_p_path=${matrix_p_path} \
            --seed=0 \
            --pos_weight=${pos_weight} \
            --log-dir="../results/${dataset}" \
            --metric_for_best_model="eval_validation_bacc" \
            --cls_head_layer=${cls_head_layer}
            
            echo "--- 实验 #${exp_count} 完成 ---"
            echo ""
            exp_count=$((exp_count + 1))
        done
    done
done

echo "🎉 所有超参数组合实验已完成！共 $((exp_count - 1)) 个实验。"

