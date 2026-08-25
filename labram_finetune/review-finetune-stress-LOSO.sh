#!/bin/bash
# =============================================================================
# Stress fine-tune launcher (17-fold subject-independent LOSO, val 選 best epoch)
# =============================================================================
# 配置对应原有 review-fietune-stress_swien.sh 里的第 4 段
# "STRESS - finetune all (全微調) - hpo - 1 層 Classifier"：
#   freeze_type=all（全微调，不冻结 backbone）, model_type=1ly（labram_base_patch200_200，
#   不是 cbramod3lyclassifier 那个 3 层头）。
# 超参数是从该 HPO 网格里选定的固定值（lr=5e-4, wd=1e-3, batch_size=1024），LOSO 17 折
# 全部沿用同一组，不逐折重新搜索。
#
# 协议与 review-finetune-Motor-LOSO.sh 一致：fold i: test=subjects[i],
# val=subjects[(i+1)%N]，val balanced_accuracy 选 best epoch，--no_save_ckpt（不落盘
# checkpoint，save_loso_fold_results 直接用内存里的 best-val-BACC 权重重新推理 test 集），
# 每折跑完写 {task}_{model}_fold{i:02d}.npz/.json 到 fold_results_dir，可断点续跑
# （json 已存在的折直接跳过）。
#
# 与 Motor LOSO 的关键差异（都已经在 utils.py / run_class_finetuning.py 里处理）：
#   - Stress 是二分类（nb_classes=1），模型输出单个 logit，save_loso_fold_results
#     走 sigmoid+0.5 阈值分支（不是 softmax+argmax），npz 里的 y_prob 仍统一存成
#     (N,2) = [1-sigmoid, sigmoid]，跟 cbramod/neurolm 的 Stress LOSO npz schema一致。
#   - Stress 17 个受试者里有 11 个只做过 increase 或 normal 单一条件，某折的 val/test
#     可能是单类别受试者，balanced_accuracy_score 在这种情况下退化为该类的 recall，
#     不会报错，选模不受影响（跟 cbramod 的处理一致，见 project 记忆 project_loso_reviewer_revision）。
#
# 数据目录：dataset_path 指向 preprocessing/stress_preprocess.ipynb 里 process_stress_dataset()
# 的 model_name="labram" 输出（Sub<NN>_ 文件名，带 subject id，供 LOSO 用）。生成方式见
# stress_data/run_labram_preprocess.sh（跑 stress_data/_run_labram_preprocess.py 处理全部
# 17 个受试者，再自动搬到 labram_finetune/augmented_data/ 下）。
#
# 用法：
#   bash review-finetune-stress-LOSO.sh   (从 labram_finetune/ 目录下运行)
# Resumable: 一折的 json 已存在就跳过。
# =============================================================================
set -e

# Repo root 是这个脚本自己所在目录（run_class_finetuning.py 跟它同级），按脚本自身
# 位置解析，谁 clone 到哪都能跑。
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=Stress
dataset_path=./augmented_data/labram_Stress_noleak_30chan_no400up_swien42
gpu_id=0

chan_size=30
epochs=50
seed=0
lr=5e-4
wd=1e-3
bs=1024

model_type=1ly
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

model_name=labram
task_name=stress

exp_name=exp_finetune_all_1ly_LOSO
parent_dir=./new_ckpt/${dataset}-LOSO/${exp_name}-${model_type}-${freeze_type}
fold_results_dir=./fold_results_labram_stress

# === 动态发现受试者列表（不写死），并按数字排序 ===
mapfile -t SUBJECTS < <(
    for split in train val test; do
        ls "${dataset_path}/${split}" 2>/dev/null
    done | grep -oE 'Sub[0-9]+' | sort -u -t b -k2 -n
)
N=${#SUBJECTS[@]}

echo "===================================================="
echo "Discovered ${N} subjects: ${SUBJECTS[*]}"
echo "===================================================="

if [ "${N}" -lt 3 ]; then
    echo "ERROR: need at least 3 subjects for subject-independent LOSO, found ${N}"
    exit 1
fi

for (( i=0; i<N; i++ )); do
    test_subject=${SUBJECTS[$i]}
    val_idx=$(( (i + 1) % N ))
    val_subject=${SUBJECTS[$val_idx]}

    fold_tag=$(printf "fold%02d_test%s" "$i" "$test_subject")
    json_out="${fold_results_dir}/${task_name}_${model_name}_fold$(printf "%02d" "$i").json"
    model_dir=${parent_dir}/${fold_tag}

    if [ -f "${json_out}" ]; then
        echo "=== [${fold_tag}] already completed (${json_out} exists), skipping ==="
        continue
    fi

    echo "=================================================================="
    echo "=== LOSO fold $((i+1))/${N}: ${fold_tag}  (val=${val_subject}, test=${test_subject}) ==="
    echo "=================================================================="

    CUDA_VISIBLE_DEVICES=${gpu_id} python run_class_finetuning.py \
        --output_dir ${model_dir} \
        --dataset_path ${dataset_path} \
        --channel_size ${chan_size} \
        ${FREEZE_ARG} \
        --finetune ./checkpoints/labram-base.pth \
        --model ${MODEL_ARG} \
        --classifier_window_size 5 \
        --abs_pos_emb \
        --dist_eval \
        --dataset ${dataset} \
        --seed ${seed} \
        --epochs ${epochs} \
        --lr ${lr} \
        --weight_decay ${wd} \
        --batch_size ${bs} \
        --min_lr 1e-6 \
        --opt_betas 0.9 0.999 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --update_freq 1 \
        --warmup_epochs 3 \
        --disable_rel_pos_bias \
        --disable_qkv_bias \
        --no_save_ckpt \
        --split_mode subject_independent \
        --test_subject ${test_subject} \
        --val_subject ${val_subject} \
        --fold_idx $i \
        --model_name ${model_name} \
        --task_name ${task_name} \
        --fold_results_dir ${fold_results_dir}
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
