#!/bin/bash
# =============================================================================
# EEGPT Stress fine-tune launcher (17-fold subject-independent LOSO, val-selected epoch)
# =============================================================================
# Mirrors review-finetune-Motion-LOSO.sh's protocol (same script, same repo),
# applied to the Stress task via linear_probe_EEGPT_Stress.py:
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = rest.
#   Best epoch picked by validation balanced accuracy (see ModelCheckpoint +
#   BestEpochTracker in linear_probe_EEGPT_Stress.py, only active when
#   --split_mode subject_independent). After each fold,
#   {task}_{model}_fold{i:02d}.npz/.json are written under fold_results_dir
#   so every downstream metric can be recomputed later without retraining
#   (see save_eegpt_stress_fold_results()).
#
# Stress only has 17 subjects (not Motion's 20), and 11 of them only ever
# recorded one condition (increase-only or normal-only), so several folds'
# val/test subject is single-class. eegpt's Stress head is already a genuine
# 2-class softmax classifier (same shape as Motion, unlike cbramod/labram/
# neurolm's single-logit+sigmoid Stress heads), so balanced_accuracy_score
# doesn't crash on a single-class val subject during training (it degrades to
# that class's recall) -- no special-casing needed in the training loop
# itself. The only place single-class folds need NaN protection is roc_auc/
# pr_auc, which are computed post-hoc from the saved npz by
# compute_metrics_from_npz.py / aggregate_loso_results_stress.py, not here.
#
# Hyperparameters are fixed (no HPO sweep): lr=1e-3, weight_decay=1e-3,
# batch_size=64, full finetune (NOT linear probe -- freeze_encoder=False,
# see LOSO_* constants at the top of linear_probe_EEGPT_Stress.py). lr/wd/bs
# are user-specified; epochs/freeze_encoder/encoder_lr_ratio follow the
# previous Stress config (grid-search defaults at the top of the file:
# max_epochs=50, freeze_encoder=False, encoder_lr_ratio=0.1). Hardcoded in
# run_loso_fold(), not passed from this script, so they can't drift between
# folds.
#
# NOTE: Stress's 5s@256Hz=1280 sample sequences are 5x longer than Motor's
# 256, so full-encoder finetune is far more memory-hungry per batch than
# Motor LOSO on the same GPU. If you hit CUDA OOM on the real sweep, lower
# LOSO_BATCH_SIZE in linear_probe_EEGPT_Stress.py (mixed precision was tried
# and reverted -- it didn't resolve the OOM in testing and added a real
# fp16-on-CPU-softmax bug, not worth the complexity here).
#
# The original linear_probe_EEGPT_Stress.py grid-search entry point
# (BS_LIST x LR_LIST x WD_LIST over the fixed train/val/test split) is
# untouched and keeps behaving exactly as before when this script's new
# --split_mode flag is not passed.
#
# Data: data_root is hardcoded in linear_probe_EEGPT_Stress.py as "Stress_data"
# (no --dataset_path CLI override, same convention as Motion's data_root=
# "./Motiondata"). Generate it with stress_data/run_eegpt_preprocess.py (see
# stress_data/run_eegpt_preprocess.sh), which moves its output to
# eegpt_finetune/Stress_data/{train,val,test}. Subject list is discovered
# dynamically from there rather than hardcoded.
#
# Usage:
#   bash review-finetune-stress-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

# Repo root for this model is one level up from this script (scripts/../),
# resolved from the script's own location so this works regardless of who
# clones it or where from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gpu_id=0
model_name=eegpt
task_name=stress
fold_results_dir=./fold_results_eegpt_stress

if [ ! -d "Stress_data" ]; then
    echo "ERROR: Stress_data/ not found under $(pwd)."
    echo "Generate it first: cd ../stress_data && bash run_eegpt_preprocess.sh"
    exit 1
fi

# === 动态发现受试者列表（不写死），并按数字排序 ===
mapfile -t SUBJECTS < <(
    for split in train val test; do
        ls "Stress_data/${split}" 2>/dev/null
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

    CUDA_VISIBLE_DEVICES=${gpu_id} python linear_probe_EEGPT_Stress.py \
        --split_mode subject_independent \
        --test_subject "${test_subject}" \
        --val_subject "${val_subject}" \
        --fold_idx "$i" \
        --model_name ${model_name} \
        --task_name ${task_name} \
        --fold_results_dir ${fold_results_dir}
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
