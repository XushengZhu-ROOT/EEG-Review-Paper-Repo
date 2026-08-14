#!/bin/bash
# =============================================================================
# Motor6Class fine-tune launcher (20-fold subject-independent LOSO,
# val-balanced-accuracy-selected checkpoint)
# =============================================================================
# Same protocol as the other models' Motion/Motor LOSO scripts (see
# cbramod_finetune/scripts/review-finetune-motortask_R2_loso.sh,
# biot_finetune/scripts/review-finetune-Motion-LOSO.sh,
# eegpt_finetune/scripts/review-finetune-Motion-LOSO.sh,
# labram_finetune/review-finetune-Motor-LOSO.sh,
# neurolm_finetune/scripts/review-finetune-motor-LOSO.sh):
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = the rest.
#   Best checkpoint picked by validation balanced_accuracy -- this reuses the
#   existing HF Trainer load_best_model_at_end / metric_for_best_model
#   machinery already wired in src/train_gpt.py (unchanged), only active when
#   --split-mode subject_independent (see prepare_motor6class_dataset_subject_independent()
#   + save_loso_fold_results() in src/train_gpt.py). After each fold,
#   {task}_{model}_fold{i:02d}.npz/.json are written under fold_results_dir so
#   every downstream metric can be recomputed later without retraining.
#
# Hyperparameters below are the LOSO-specific fixed config (lr=3e-4,
# weight_decay=1e-2, batch_size=128); everything else (ft_only_encoder=True,
# cls_head_layer=3ly, chunk/encoder settings, training_steps) reuses the
# existing "author config" full fine-tune setup for motor6class
# (see review-finetune-motor6class-author-full.sh).
#
# Subject list is discovered dynamically from AllSubjects_Epochs rather than
# hardcoded, so it can't silently go stale if the subject roster changes.
#
# Usage:
#   bash review-finetune-motor6class-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

# Repo root is this script's own parent directory (src/train_gpt.py lives
# alongside it), resolved from the script's own location so this works
# regardless of who clones it or where from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dataset=motor6class
dataset_path="./AllSubjects_Epochs"
matrix_p_path="./tMatrix_22x20_motor.npy"
gpu_id=0
pos_weight=-1.0  # 6分类任务，不需要pos_weight

# 数据预处理参数（沿用 author config）
num_chunks=2
chunk_len=250  # 1秒 = 250个时间点 @ 250Hz（与预训练模型匹配）
chunk_ovlp=0   # 不重叠

# 模型参数（LOSO 专用超参：lr=3e-4, wd=1e-2, bs=128）
cls_head_layer='3ly'
lr=3e-4
wd=1e-2
batch_size=128

model_name=neurogpt
task_name=motor

exp_name="LOSO_full_3ly_lr${lr}_wd${wd}_bs${batch_size}"
fold_results_dir="./fold_results_neurogpt"

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

    if [ -f "${json_out}" ]; then
        echo "=== [${fold_tag}] already completed (${json_out} exists), skipping ==="
        continue
    fi

    echo "=================================================================="
    echo "=== LOSO fold $((i+1))/${N}: ${fold_tag}  (val=${val_subject}, test=${test_subject}) ==="
    echo "=================================================================="

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 src/train_gpt.py \
    --training-style='decoding' \
    --num-decoding-classes=6 \
    --training-steps=10000 \
    --eval_every_n_steps=500 \
    --log-every-n-steps=500 \
    --num_chunks=${num_chunks} \
    --per-device-training-batch-size=${batch_size} \
    --per-device-validation-batch-size=${batch_size} \
    --chunk_len=${chunk_len} \
    --chunk_ovlp=${chunk_ovlp} \
    --run-name=${exp_name}_${fold_tag} \
    --ft-only-encoder='True' \
    --fold_i=$i \
    --num-encoder-layers=6 \
    --num-hidden-layers=6 \
    --learning-rate=${lr} \
    --weight-decay=${wd} \
    --use-encoder='True' \
    --embedding-dim=1024 \
    --pretrained-model='./pytorch_model.bin' \
    --dataset-name=${dataset} \
    --dst-data-path=${dataset_path} \
    --matrix_p_path=${matrix_p_path} \
    --seed=0 \
    --pos_weight=${pos_weight} \
    --log-dir="./results/${dataset}_LOSO" \
    --metric_for_best_model="eval_validation_bacc" \
    --cls_head_layer=${cls_head_layer} \
    --split-mode=subject_independent \
    --test-subject=${test_subject} \
    --val-subject=${val_subject} \
    --fold-idx=$i \
    --model-name=${model_name} \
    --task-name=${task_name} \
    --fold-results-dir=${fold_results_dir}

    echo "--- fold ${fold_tag} 完成 ---"
    echo ""
done

echo "All LOSO folds done. npz/json results under: ${fold_results_dir}"
