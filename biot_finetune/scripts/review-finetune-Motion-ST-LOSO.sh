#!/bin/bash
# =============================================================================
# Motion-ST fine-tune launcher (20-fold subject-independent LOSO, val-selected epoch)
# =============================================================================
# ST-transformer counterpart of review-finetune-Motion-LOSO.sh (the BIOT
# script). Same protocol, same run_multiclass_supervised.py entry point:
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = rest.
#   Best epoch picked by validation BACC (ModelCheckpoint + BestEpochTracker,
#   only active when --split_mode subject_independent). After each fold,
#   {task}_{model}_fold{i:02d}.npz/.json are written under fold_results_dir
#   so every downstream metric can be recomputed later without retraining.
#
# Differences from the BIOT script (see prepare_Motion_dataloader() /
# MotionSTLoader in utils.py):
#   - dataset=Motion-ST reads from Motiondata_ST (native 20ch @ 250Hz),
#     not AllSubjects_Epochs (BIOT's 200Hz data, hard-reduced to 16ch to
#     match the pretrained checkpoint).
#   - STTransformer has no pretrained backbone, so there's no
#     pretrain_model_path / channel-matching concern: in_channels=20 uses
#     all native channels, matching how Sleep/SEED already feed
#     STTransformer their full channel count.
#   - sample_length=1 (Motiondata_ST epochs are 1s @ 250Hz = 250 samples;
#     see datamake/movement_st.py's epoch_duration=1.0), not the 3s used
#     by the BIOT Motion script.
#   - lr=1e-3, wd=1e-3, bs=512.
#
# The BIOT script/pipeline is untouched by this file.
#
# Usage:
#   bash review-finetune-Motion-ST-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

# Repo root is one level up from this script (scripts/../), resolved from
# the script's own location so this works regardless of who clones it or
# where to.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dataset=Motion-ST
gpu_id=0

dataset_channels=20
in_channels=20
sample_length=1
sampling_rate=250
epochs=50

classifier_type=STTransformer
model_name=st
task_name=motion

lr=0.001
wd=0.001
bs=512

fold_results_dir=./fold_results_st

# === 动态发现受试者列表（不写死），并按数字排序 ===
mapfile -t SUBJECTS < <(
    for split in train val test; do
        ls "Motiondata_ST/${split}" 2>/dev/null
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

    CUDA_VISIBLE_DEVICES=${gpu_id} python run_multiclass_supervised.py \
        --exp_name "exp_loso_${fold_tag}" \
        --dataset ${dataset} \
        --n_classes 6 \
        --dataset_channels ${dataset_channels} \
        --in_channels ${in_channels} \
        --sampling_rate ${sampling_rate} \
        --sample_length ${sample_length} \
        --batch_size ${bs} \
        --lr ${lr} \
        --weight_decay ${wd} \
        --epochs ${epochs} \
        --model ${classifier_type} \
        --split_mode subject_independent \
        --test_subject ${test_subject} \
        --val_subject ${val_subject} \
        --fold_idx $i \
        --model_name ${model_name} \
        --task_name ${task_name} \
        --fold_results_dir ${fold_results_dir}
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
