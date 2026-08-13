#!/bin/bash
# 测试实验完成检查函数
# 用法: ./test_check_completed.sh

dataset=motor6class
MAX_STEPS=10000

# 检查实验是否完成的函数
check_experiment_completed() {
    local exp_name=$1
    local exp_dir="../results/${dataset}/${exp_name}"
    
    echo "=========================================="
    echo "检查实验: ${exp_name}"
    echo "实验目录: ${exp_dir}"
    
    # 检查实验目录是否存在
    if [ ! -d "$exp_dir" ]; then
        echo "❌ 实验目录不存在"
        return 1
    fi
    echo "✅ 实验目录存在"
    
    # 检查train_history.csv是否存在
    local train_history="${exp_dir}/train_history.csv"
    if [ ! -f "$train_history" ]; then
        echo "❌ train_history.csv 不存在"
        return 1
    fi
    echo "✅ train_history.csv 存在"
    
    # 显示文件内容（前5行和后5行）
    echo "--- train_history.csv 前5行 ---"
    head -n 5 "$train_history" 2>/dev/null || echo "无法读取"
    echo "--- train_history.csv 后5行 ---"
    tail -n 5 "$train_history" 2>/dev/null || echo "无法读取"
    
    # 检查训练是否达到max_steps（读取最后一行，提取step值）
    local last_line=$(tail -n 1 "$train_history" 2>/dev/null)
    echo "最后一行: ${last_line}"
    
    local last_step=$(echo "$last_line" | cut -d',' -f1)
    echo "提取的step值: '${last_step}'"
    
    # 检查step是否为数字
    if ! [[ "$last_step" =~ ^[0-9]+$ ]]; then
        echo "❌ step值不是数字: '${last_step}'"
        return 1
    fi
    
    echo "当前训练到step: ${last_step}/${MAX_STEPS}"
    
    if [ "$last_step" -lt "$MAX_STEPS" ]; then
        echo "❌ 训练未达到max_steps (${last_step} < ${MAX_STEPS})"
        return 1
    fi
    echo "✅ 训练已达到max_steps"
    
    # 检查测试集预测文件是否存在（说明测试集预测已完成）
    local test_predictions="${exp_dir}/test_predictions.npy"
    local test_labels="${exp_dir}/test_label_ids.npy"
    
    if [ ! -f "$test_predictions" ]; then
        echo "❌ test_predictions.npy 不存在"
        return 1
    fi
    echo "✅ test_predictions.npy 存在"
    
    if [ ! -f "$test_labels" ]; then
        echo "❌ test_label_ids.npy 不存在"
        return 1
    fi
    echo "✅ test_label_ids.npy 存在"
    
    # 所有检查都通过，实验已完成
    echo "=========================================="
    echo "✅✅✅ 实验已完成！应该跳过 ✅✅✅"
    echo "=========================================="
    return 0
}

# 测试所有实验
echo "=========================================="
echo "开始测试所有实验..."
echo "=========================================="
echo ""

# 自动检测所有hpo_exp开头的实验目录
results_dir="../results/${dataset}"
if [ ! -d "$results_dir" ]; then
    echo "错误: 结果目录不存在: $results_dir"
    exit 1
fi

# 获取所有hpo_exp实验目录
exp_dirs=$(ls -d "${results_dir}"/hpo_exp* 2>/dev/null | sort -V)

if [ -z "$exp_dirs" ]; then
    echo "未找到任何hpo_exp实验目录"
    exit 1
fi

echo "找到以下实验目录:"
echo "$exp_dirs" | while read dir; do
    echo "  - $(basename "$dir")"
done
echo ""

# 测试每个实验
for exp_dir in $exp_dirs; do
    exp_name=$(basename "$exp_dir")
    echo ""
    echo "##########################################"
    echo "测试实验: $exp_name"
    echo "##########################################"
    
    if check_experiment_completed "$exp_name"; then
        echo ""
        echo "🎯 最终结论: ✅ 跳过实验 (${exp_name})"
    else
        echo ""
        echo "🎯 最终结论: ❌ 需要运行实验 (${exp_name})"
    fi
    echo ""
done

echo "=========================================="
echo "测试完成！"
echo "=========================================="

