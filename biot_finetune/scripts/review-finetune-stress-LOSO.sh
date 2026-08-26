#!/bin/bash
# =============================================================================
# Stress fine-tune launcher (subject-independent LOSO, one fold per subject,
# val-selected epoch)
# =============================================================================
# Mirrors review-finetune-Motion-LOSO.sh's protocol, applied to the Stress
# task (binary: 0=normal, 1=increase):
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = rest.
#   Best epoch picked by validation balanced_accuracy (see ModelCheckpoint +
#   BestEpochTracker in run_binary_supervised.py, only active when
#   --split_mode subject_independent). After each fold, {task}_{model}_fold{i:02d}
#   .npz/.json are written under fold_results_dir so every downstream metric
#   can be recomputed later without retraining (see save_stress_fold_results()).
#
# Unlike Motion (6-class softmax), Stress is binary with a single sigmoid
# logit, and ~11 of the ~17 subjects only ever recorded one condition
# (increase-only or normal-only -- see stress_data/subject_edf_mapping.csv),
# so many folds' val/test subject is single-class. run_binary_supervised.py's
# _compute_binary_metrics()/binary_metrics_fn() handle this: accuracy/
# balanced_accuracy are computed normally (well-defined for single-class
# y_true), only roc_auc/pr_auc are NaN-guarded when undefined -- same
# treatment as cbramod_finetune/finetune_evaluator.py's
# get_metrics_for_binaryclass. Use aggregate_loso_results_stress.py's POOLED
# section for the numbers to report (per-fold Kappa/ROC-AUC/PR-AUC are
# NaN-polluted by design).
#
# Classifier is LabramClassifier-BIOT (Labram_style_BIOTClassifier /
# Labram_style_Ada_BIOT), i.e. the 1-layer LayerNorm+Linear head, NOT the
# 3-layer CBraMod_3lyStyle_LayerNorm-BIOT head used for Motion -- matches the
# original (pre-LOSO) review-finetune-stress.sh's "1 層 Classifier" config.
#
# Hyperparameters (lr/wd/bs) are the user-specified LOSO config; everything
# else (sample_length/epochs/pretrain/channels/sampling_rate/token_size/
# hop_length/freeze_type=all i.e. full finetune) is carried over unchanged
# from the original review-finetune-stress.sh's author-config block.
#
# Data: Stress_data/{train,val,test} (200Hz, no filtering -- see
# stress_data/_run_biot_preprocess.py's preprocess_stress_biot / run
# stress_data/run_biot_preprocess.sh to generate it). Subject list is
# discovered dynamically rather than hardcoded, same reasoning as
# review-finetune-Motion-LOSO.sh.
#
# Usage:
#   bash scripts/review-finetune-stress-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dataset=Stress
gpu_id=0

dataset_channels=30
sample_length=5
epochs=50
pretrain_model_channels=16
pretrain_path=pretrained-models/EEG-PREST-${pretrain_model_channels}-channels.ckpt

classifier_type=LabramClassifier-BIOT
model_name=biot
task_name=stress

lr=0.001
wd=0.001
bs=512

freeze_type=all
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi

fold_results_dir=./fold_results_biot_stress

# === 动态发现受试者列表（不写死），并按数字排序 ===
mapfile -t SUBJECTS < <(
    for split in train val test; do
        ls "${dataset}_data/${split}" 2>/dev/null
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

    CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
        --exp_name "exp_loso_${fold_tag}" \
        --dataset ${dataset} \
        --n_classes 1 \
        --dataset_channels ${dataset_channels} \
        --in_channels ${pretrain_model_channels} \
        --sampling_rate 200 \
        --token_size 200 \
        --hop_length 100 \
        --sample_length ${sample_length} \
        --batch_size ${bs} \
        --lr ${lr} \
        --weight_decay ${wd} \
        --epochs ${epochs} \
        --model ${classifier_type} \
        --pretrain_model_path ${pretrain_path} \
        --output_dir StressLOSO-${model_name} \
        --split_mode subject_independent \
        --test_subject ${test_subject} \
        --val_subject ${val_subject} \
        --fold_idx $i \
        --model_name ${model_name} \
        --task_name ${task_name} \
        --fold_results_dir ${fold_results_dir} \
        ${FREEZE_ARG}
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
