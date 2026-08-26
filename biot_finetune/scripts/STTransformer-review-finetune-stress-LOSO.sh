#!/bin/bash
# =============================================================================
# Stress fine-tune launcher for STTransformer (subject-independent LOSO, one
# fold per subject, val-selected epoch)
# =============================================================================
# ST-transformer counterpart of review-finetune-stress-LOSO.sh (the BIOT
# script). Same protocol, same run_binary_supervised.py entry point:
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = rest.
#   Best epoch picked by validation balanced_accuracy (ModelCheckpoint +
#   BestEpochTracker, only active when --split_mode subject_independent).
#   After each fold, {task}_{model}_fold{i:02d}.npz/.json are written under
#   fold_results_dir so every downstream metric can be recomputed later
#   without retraining.
#
# Differences from the BIOT script (see prepare_Stress_dataloader() in
# utils.py / run_binary_supervised.py):
#   - dataset=Stress-ST reads from Stress_data_ST (250Hz, 4-40Hz band-pass --
#     see stress_data/_run_st_preprocess.py's preprocess_stress_sttransformer),
#     not Stress_data (BIOT's 200Hz, unfiltered data) -- same "ST uses its own
#     sampling rate/filtering, doesn't share BIOT's preprocessing" convention
#     as Motion vs Motion-ST.
#   - STTransformer has no pretrained backbone and no channel-count
#     constraint, so in_channels=30 uses all native channels directly (no
#     chan_conv adaptation needed, unlike BIOT's dataset_channels=30 ->
#     in_channels=16).
#   - Classifier head is STTransformer's own (ELU + single Linear -- already
#     1-layer, there is no separate 3-layer STTransformer variant in this
#     repo, so no head selection needed here unlike BIOT's
#     LabramClassifier-BIOT vs CBraMod_3lyStyle_LayerNorm-BIOT choice).
#   - lr=5e-3, wd=1e-5, bs=256 (user-specified LOSO config); sample_length/
#     epochs carried over from the original (pre-LOSO)
#     STTransformer-review-finetune-stress.sh.
#
# ~11 of the ~17 subjects only ever recorded one condition (increase-only or
# normal-only), so many folds' val/test subject is single-class -- handled by
# the same NaN-guard in run_binary_supervised.py as the BIOT script (see its
# comments for detail). Use aggregate_loso_results_stress.py's POOLED section
# for the numbers to report.
#
# The BIOT script/pipeline is untouched by this file.
#
# Usage:
#   bash scripts/STTransformer-review-finetune-stress-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dataset=Stress-ST
gpu_id=0

dataset_channels=30
in_channels=30
sample_length=5
epochs=50
sampling_rate=250

classifier_type=STTransformer
model_name=st
task_name=stress

lr=0.005
wd=0.00001
bs=256

fold_results_dir=./fold_results_st_stress

# === 动态发现受试者列表（不写死），并按数字排序 ===
mapfile -t SUBJECTS < <(
    for split in train val test; do
        ls "Stress_data_ST/${split}" 2>/dev/null
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
        --in_channels ${in_channels} \
        --sampling_rate ${sampling_rate} \
        --sample_length ${sample_length} \
        --batch_size ${bs} \
        --lr ${lr} \
        --weight_decay ${wd} \
        --epochs ${epochs} \
        --model ${classifier_type} \
        --output_dir StressLOSO-${model_name} \
        --split_mode subject_independent \
        --test_subject ${test_subject} \
        --val_subject ${val_subject} \
        --fold_idx $i \
        --model_name ${model_name} \
        --task_name ${task_name} \
        --fold_results_dir ${fold_results_dir}
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
