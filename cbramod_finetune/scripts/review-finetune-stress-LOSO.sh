#!/bin/bash
# =============================================================================
# Stress fine-tune launcher (17-fold subject-independent LOSO, 有验证集选模)
# =============================================================================
# 协议与 review-finetune-motortask_R2_loso.sh 完全一致：
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = 其余
#   选模标准：验证集 BACC 最高的 epoch（finetune_trainer.py train_for_binaryclass
#   里已有的逻辑，未改动）。每折训练结束后用被选中的最佳 epoch 权重对 test 集
#   重新推理一次，额外保存 {task}_{model}_fold{i:02d}.npz / .json
#   （sample_id 事后可复现指标用，跟 Motor 用的是同一套 aggregate_loso_results.py）。
#
# 注意：stress 17 个受试者里有 11 个只做过 increase 或只做过 normal 单一条件
# （不像 Motor 每个受试者都有全部 6 类），当某折的 val/test 受试者恰好是单一
# 类别时，该折的 val_roc_auc/test_roc_auc/val_pr_auc/test_pr_auc 会是 NaN
# （finetune_evaluator.py 已改为优雅处理，不会崩溃/不影响 val_bacc 选模）。
#
# 固定超参数（lr=5e-4, wd=2e-3, bs=64）取自 review-finetune-stress.sh 里
# "STRESS - finetune (全微調) - hpo - 1 層 Classifier" 那组网格中的一组，
# 其余配置（classifier/epochs/seed/channel_size/window_size/freeze_type=all）
# 与该脚本的 finetune-all 段保持一致。
#
# 数据目录：dataset_path 指向 preprocessing/stress_preprocess.ipynb 新版本
# 输出的 cbramod_Stress_noleak_30chan_no400up_swien42（Sub<NN>_ 文件名，
# 带 subject id），不是旧的 Stress_noleak_30chan_no400up_seed_siwen42（无
# subject id，无法做 LOSO）。已经放在 cbramod_finetune/augmented_data/ 下了
# （8334 个 pickle，train/val/test = 6630/858/846），本地路径就是这个相对路径；
# 上服务器如果目录结构不同，改这一行就行。
#
# 用法：
#   bash scripts/review-finetune-stress-LOSO.sh   (从 cbramod_finetune/ 目录下运行)
# 支持断点续跑：某折的 model_dir 下已有 training_summary.json 就跳过该折。
# =============================================================================
set -e

dataset=CustomStress
dataset_path=./augmented_data/cbramod_Stress_noleak_30chan_no400up_swien42
exp_name=exp_hpo_lr0.0005_wd0.002_bs64_LOSO
gpu_id=0
chan_size=30
window_size=5
epochs=50
lr=0.0005
wd=0.002
bs=64
classifier=Labram_style_classifier
seed=0
freeze_type=all
model_name=cbramod
task_name=stress

# 17 个受试者，排序固定（与 datasets/custom_stress_dataset.py 里
# sorted(subject_to_files.keys()) 的顺序一致；没有 Sub15，因为
# subject_edf_mapping.csv 里没有 Patient_ID=15）
SUBJECTS=(Sub01 Sub02 Sub03 Sub04 Sub05 Sub06 Sub07 Sub08 Sub09 Sub10 Sub11 \
          Sub12 Sub13 Sub14 Sub16 Sub17 Sub18)
N=${#SUBJECTS[@]}   # 17

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--frozen"
else
    FREEZE_ARG=""
fi

parent_dir=./models_weights/${dataset}/${exp_name}-${classifier}-${freeze_type}-subject_independent
fold_results_dir=./fold_results_stress

for (( i=0; i<N; i++ )); do
    test_subject=${SUBJECTS[$i]}
    val_idx=$(( (i + 1) % N ))
    val_subject=${SUBJECTS[$val_idx]}

    fold_tag=$(printf "fold%02d_test%s" "$i" "$test_subject")
    model_dir=${parent_dir}/${fold_tag}

    if [ -f "${model_dir}/training_summary.json" ]; then
        echo "=== [${fold_tag}] already completed, skipping (val=${val_subject}, test=${test_subject}) ==="
        continue
    fi

    echo "=================================================================="
    echo "=== LOSO fold $((i+1))/${N}: ${fold_tag}  (val=${val_subject}, test=${test_subject}) ==="
    echo "=================================================================="

    python finetune_main.py \
        --seed ${seed} \
        --epochs ${epochs} \
        --downstream_dataset ${dataset} \
        --datasets_dir ${dataset_path} \
        --optimizer AdamW \
        --num_of_classes 2 \
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
        --split_mode subject_independent \
        --val_subject ${val_subject} \
        --test_subject ${test_subject} \
        --fold_idx ${i} \
        --model_name ${model_name} \
        --task_name ${task_name} \
        --fold_results_dir ${fold_results_dir}
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
echo "Aggregate BACC/kappa/F1 with: python3 aggregate_loso_results.py --fold_results_dir ${fold_results_dir} --task ${task_name} --model ${model_name} --n_classes 2 --out loso_results_${model_name}_${task_name}.csv --cm_out loso_confusion_matrix_${model_name}_${task_name}.npy"
