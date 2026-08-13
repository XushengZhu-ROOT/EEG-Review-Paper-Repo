#!/bin/bash
# =============================================================================
# MotorTask fine-tune launcher (R2: 20-fold subject-independent LOSO, 有验证集选模)
# =============================================================================
# 老师会后修改（相对上一版）：
#   1. 折的规则改为 test = subjects[i], val = subjects[(i+1) % N], train = 其余，
#      选模标准改回"验证集 BACC 最高的 epoch"（不是固定最后一轮）。
#      这条路径完全复用 R1 就有的 --val_subject/--test_subject 机制
#      （见 datasets/motortask_dataset.py 的 _make_subject_fold_split），
#      不传 --no_val_subject（默认 False），has_val=True，
#      finetune_trainer.py 里选模逻辑和 R1 时完全一样，没有新增分支要改。
#   2. 每折训练结束后，用被选中的最佳 epoch 权重对 test 集重新推理一次，
#      额外保存 {task}_{model}_fold{i:02d}.npz / .json（sample_id 事后可复现指标用），
#      见 finetune_trainer.py 的 save_fold_results()。
#
# 注意：这一版和之前"无验证集、固定末轮"那版（exp_author_config_R2_LOSO_noval）
# 是两个不同协议，互不兼容、不能混用同一批已跑完的折。之前 _noval 版本下已经
# 跑完的 2 折（fold00 test=Sub01, fold01 test=Sub02）不适用于本协议，会在下面
# 这个新的 exp_name（带 _val 后缀）目录下重新跑，不会覆盖 _noval 那份旧结果。
#
# 折的规则（20 个受试者，排序后 subjects[0..19] = Sub01..Sub11, Sub13..Sub21）：
#   fold i:  test = subjects[i]
#            val  = subjects[(i+1) % N]   （排序中的下一个，wrap-around）
#            train = 其余 18 人
#
# 用法：
#   bash review-finetune-motortask_R2_loso.sh
# 支持断点续跑：某折的 model_dir 下已有 training_summary.json 就跳过该折。
# =============================================================================

dataset=MotorTask
dataset_path=./AllSubjects_Epochs
exp_name=exp_author_config_R2_LOSO_val
gpu_id=0
chan_size=20
window_size=1
epochs=50
lr=0.0005
wd=0.002
bs=256
classifier=all_patch_reps
seed=0
freeze_type=all
model_name=cbramod
task_name=motortask

# 20 个受试者，排序固定（与 datasets/motortask_dataset.py 里 sorted(subjects) 的顺序一致）
SUBJECTS=(Sub01 Sub02 Sub03 Sub04 Sub05 Sub06 Sub07 Sub08 Sub09 Sub10 Sub11 \
          Sub13 Sub14 Sub15 Sub16 Sub17 Sub18 Sub19 Sub20 Sub21)
N=${#SUBJECTS[@]}   # 20

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--frozen"
else
    FREEZE_ARG=""
fi

parent_dir=./models_weights/${dataset}/${exp_name}-${classifier}-${freeze_type}-subject_independent
fold_results_dir=./fold_results

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
        --split_mode subject_independent \
        --single_fold_debug True \
        --val_subject ${val_subject} \
        --test_subject ${test_subject} \
        --fold_idx ${i} \
        --model_name ${model_name} \
        --task_name ${task_name} \
        --fold_results_dir ${fold_results_dir}
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
echo "Aggregate BACC/kappa/F1 with: python3 aggregate_loso_results_R2.py --parent_dir ${parent_dir}"
