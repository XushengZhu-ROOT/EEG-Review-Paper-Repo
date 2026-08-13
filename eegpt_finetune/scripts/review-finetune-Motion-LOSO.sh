#!/bin/bash
# =============================================================================
# EEGPT Motion fine-tune launcher (20-fold subject-independent LOSO, val-selected epoch)
# =============================================================================
# Mirrors cbramod/review-finetune-motortask_R2_loso.sh and
# biot/review-finetune-Motion-LOSO.sh's protocol, applied to the EEGPT-side
# training script:
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = rest.
#   Best epoch picked by validation balanced accuracy (see ModelCheckpoint +
#   BestEpochTracker in linear_probe_EEGPT_Motor.py, only active when
#   --split_mode subject_independent). After each fold,
#   {task}_{model}_fold{i:02d}.npz/.json are written under fold_results_dir
#   so every downstream metric can be recomputed later without retraining
#   (see save_eegpt_fold_results()).
#
# Hyperparameters are fixed (no HPO sweep), matching what was agreed for this
# LOSO run: lr=1e-3, weight_decay=1e-2, batch_size=32, full finetune (NOT
# linear probe — freeze_encoder=False, see LOSO_* constants at the top of
# linear_probe_EEGPT_Motor.py). These are hardcoded in run_loso_fold(), not
# passed from this script, so they can't drift between folds.
#
# The original linear_probe_EEGPT_Motor.py grid-search entry point
# (BS_LIST x LR_LIST x WD_LIST over the fixed train/val/test split) is
# untouched and keeps behaving exactly as before when this script's new
# --split_mode flag is not passed.
#
# Subject list is discovered dynamically from Motiondata rather than
# hardcoded, so it can't silently go stale if the data changes.
#
# Usage:
#   bash review-finetune-Motion-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

# Repo root for this model is one level up from this script (scripts/../),
# resolved from the script's own location so this works regardless of who
# clones it or where from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gpu_id=0
model_name=eegpt
task_name=motion
fold_results_dir=./fold_results_eegpt

# === 动态发现受试者列表（不写死），并按数字排序 ===
mapfile -t SUBJECTS < <(
    for split in train val test; do
        ls "Motiondata/${split}" 2>/dev/null
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

    if [ -f "${json_out}" ]; then
        echo "=== [${fold_tag}] already completed (${json_out} exists), skipping ==="
        continue
    fi

    echo "=================================================================="
    echo "=== LOSO fold $((i+1))/${N}: ${fold_tag}  (val=${val_subject}, test=${test_subject}) ==="
    echo "=================================================================="

    CUDA_VISIBLE_DEVICES=${gpu_id} python linear_probe_EEGPT_Motor.py \
        --split_mode subject_independent \
        --test_subject "${test_subject}" \
        --val_subject "${val_subject}" \
        --fold_idx "$i" \
        --model_name ${model_name} \
        --task_name ${task_name} \
        --fold_results_dir ${fold_results_dir}
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
