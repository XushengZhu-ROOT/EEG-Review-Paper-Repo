#!/bin/bash
# =============================================================================
# Motion fine-tune launcher (20-fold subject-independent LOSO, val-selected epoch)
# =============================================================================
# Mirrors cbramod/review-finetune-motortask_R2_loso.sh's protocol, applied to
# the BIOT-side training script:
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = rest.
#   Best epoch picked by validation BACC (see ModelCheckpoint + BestEpochTracker
#   in run_multiclass_supervised.py, only active when --split_mode
#   subject_independent). After each fold, {task}_{model}_fold{i:02d}.npz/.json
#   are written under fold_results_dir so every downstream metric can be
#   recomputed later without retraining (see save_motion_fold_results()).
#
# This script only exercises the "author config" hyperparameters (matching
# review-finetune-Motion.sh's single run, not its HPO sweep). The original
# review-finetune-Motion.sh is untouched and keeps behaving exactly as before.
#
# Subject list is discovered dynamically from AllSubjects_Epochs rather than
# hardcoded, so a future data fix (like the Sub04 20ch fix that started this)
# can't silently go stale here.
#
# Usage:
#   bash review-finetune-Motion-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

# Repo root is one level up from this script (scripts/../), resolved from
# the script's own location so this works regardless of who clones it or
# where to.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dataset=Motion
gpu_id=0

dataset_channels=16
sample_length=3
epochs=50
pretrain_model_channels=16
pretrain_path=pretrained-models/EEG-PREST-${pretrain_model_channels}-channels.ckpt

classifier_type=CBraMod_3lyStyle_LayerNorm-BIOT
model_name=biot
task_name=motion

lr=0.0005
wd=0.00001
bs=512
pos_weight=0.413

freeze_type=all

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi

fold_results_dir=./fold_results_biot

# === 动态发现受试者列表（不写死），并按数字排序 ===
mapfile -t SUBJECTS < <(
    for split in train val test; do
        ls "AllSubjects_Epochs/${split}" 2>/dev/null
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
        --in_channels ${pretrain_model_channels} \
        --sampling_rate 200 \
        --token_size 200 \
        --hop_length 100 \
        --sample_length ${sample_length} \
        --batch_size ${bs} \
        --lr ${lr} \
        --weight_decay ${wd} \
        --pos_weight ${pos_weight} \
        --epochs ${epochs} \
        --model ${classifier_type} \
        --pretrain_model_path ${pretrain_path} \
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
