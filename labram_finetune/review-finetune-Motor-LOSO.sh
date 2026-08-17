#!/bin/bash
# =============================================================================
# Motor fine-tune launcher (20-fold subject-independent LOSO, val-selected epoch)
# =============================================================================
# Same protocol as the other models' Motion/Motor LOSO scripts (see
# cbramod_finetune/scripts/review-finetune-motortask_R2_loso.sh,
# biot_finetune/scripts/review-finetune-Motion-LOSO.sh,
# eegpt_finetune/scripts/review-finetune-Motion-LOSO.sh):
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = the rest.
#   Best epoch picked by validation balanced_accuracy (see the loso_mode
#   bookkeeping + save_loso_fold_results() in run_class_finetuning.py, only
#   active when --split_mode subject_independent). After each fold,
#   {task}_{model}_fold{i:02d}.npz/.json are written under fold_results_dir
#   so every downstream metric can be recomputed later without retraining.
#
# Hyperparameters below are the LOSO-specific fixed config (lr=9e-4,
# weight_decay=1e-4, batch_size=256); everything else (epochs, optimizer
# settings, model arch, classifier) reuses the existing "author config"
# for Motor (see review-fietune-Motor_author.sh).
#
# --no_save_ckpt: no .pth checkpoint files are written under output_dir for
# any fold. This is safe -- save_loso_fold_results() re-infers the test set
# from the best-val-BACC weights already held in memory (see run_class_finetuning.py),
# it never reloads from a saved checkpoint. Trade-off: a fold killed mid-training
# restarts from epoch 0 instead of resuming (fold-level resume via the
# {task}_{model}_fold{i:02d}.json check below still works fine).
#
# Subject list is discovered dynamically from AllSubjects_Epochs rather than
# hardcoded, so it can't silently go stale if the subject roster changes.
#
# Usage:
#   bash review-finetune-Motor-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

# Repo root is this script's own directory (run_class_finetuning.py lives
# alongside it), resolved from the script's own location so this works
# regardless of who clones it or where from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=Motor
dataset_path=./AllSubjects_Epochs
gpu_id=0

chan_size=20
epochs=50
seed=0
lr=9e-4
wd=1e-4
bs=256
pos_weight=1.0

model_type=3ly
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
task_name=motor

exp_name=exp_author_config_LOSO
parent_dir=./new_ckpt/${dataset}-posWeight${pos_weight}-LOSO/${exp_name}-${model_type}-${freeze_type}
fold_results_dir=./fold_results_labram

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
        --pos_weight ${pos_weight} \
        ${FREEZE_ARG} \
        --finetune ./checkpoints/labram-base.pth \
        --model ${MODEL_ARG} \
        --classifier_window_size 1 \
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
