#!/bin/bash
# =============================================================================
# Stress fine-tune launcher (17-fold subject-independent LOSO,
# val-balanced-accuracy-selected checkpoint)
# =============================================================================
# Same protocol as the other models' Stress LOSO scripts (see
# cbramod_finetune/scripts/review-finetune-stress-LOSO.sh,
# neurolm_finetune/scripts/review-finetune-stress-LOSO.sh,
# labram_finetune/scripts/review-finetune-Stress-LOSO.sh,
# eegpt_finetune/scripts/review-finetune-stress-LOSO.sh) and this same
# repo's own review-finetune-motor6class-LOSO.sh:
#   fold i: test = subjects[i], val = subjects[(i+1) % N], train = the rest.
#   Best checkpoint picked by validation balanced_accuracy -- reuses the
#   existing HF Trainer load_best_model_at_end / metric_for_best_model
#   machinery already wired in src/train_gpt.py (unchanged), only active when
#   --split-mode subject_independent (see
#   prepare_stress_dataset_subject_independent() + save_loso_fold_results()
#   in src/train_gpt.py). After each fold, {task}_{model}_fold{i:02d}.npz/.json
#   are written under fold_results_dir so every downstream metric can be
#   recomputed later without retraining.
#
# Stress's decoding head outputs 2 logits (num-decoding-classes=2, softmax +
# CrossEntropyLoss via decoder/gpt.py's add_decoding_head -- same shape as
# eegpt's Stress head, NOT the single-logit+sigmoid shape used by
# cbramod/labram/neurolm), so val/test balanced_accuracy is well-defined even
# for a single-class val/test subject (11/17 stress subjects only ever
# recorded one condition -- see stress_data/subject_edf_mapping.csv). The
# existing make_decoding_accuracy_metrics() in src/trainer/make.py already
# guards roc_auc (try/except -> NaN on single-class truths) for the binary
# branch, so no metrics-code change was needed for this -- same handling as
# cbramod's finetune_evaluator.py NaN-guard for roc_auc/pr_auc on single-class
# folds.
#
# Classification head: --cls_head_layer=1ly (NOT 3ly) -- Stress uses a single
# Linear layer on top of the pooled encoder output (EEGConformer's built-in
# "1ly" cls_head_layer, see encoder/conformer_braindecode.py), matching the
# repo-wide "Stress uses a 1-layer head" convention (see eegpt/labram's Stress
# classifiers) and neurogpt's own author config for Stress
# (scripts/review-finetune-stress.sh already used cls_head_layer='1ly').
#
# Hyperparameters: lr=3e-4, weight_decay=1e-1, batch_size=64 (user-specified
# for this LOSO sweep). Everything else reuses neurogpt's own "author config"
# for Stress (scripts/review-finetune-stress.sh): ft_only_encoder=True,
# num_encoder_layers=6, num_hidden_layers=6, embedding_dim=1024,
# training_steps=10000, eval/log_every_n_steps=500, pos_weight=-1.0 (no class
# imbalance weighting), use_encoder=True.
#
# num_chunks=2/chunk_ovlp=0 unchanged from the author config ("5秒钟切成2个chunk");
# chunk_len is rescaled from 500 (200Hz author config) to 625 so the same
# 2.5s-per-chunk split covers the new 250Hz LOSO data
# (stress_data/_run_neurogpt_preprocess.py's preprocess_stress_neurogpt
# resamples to 250Hz to match neurogpt's own Motor6Class convention, not the
# author config's original 200Hz augmented_data source).
#
# Data: dataset_path points at ./Stress_data (produced by
# stress_data/run_neurogpt_preprocess.sh moving
# stress_data/augmented_data/neurogpt_Stress_noleak_30chan_no400up_swien42/
# here), matrix_p_path at ./tMatrix_22x30_stress.npy (generated via
# channel_matrix/stress.py, same MNE forward-solution pinv approach as
# channel_matrix/motor.py's tMatrix_22x20_motor.npy).
#
# Subject list is discovered dynamically from Stress_data rather than
# hardcoded, so it can't silently go stale if the subject roster changes.
#
# Usage:
#   bash scripts/review-finetune-stress-LOSO.sh
# Resumable: a fold is skipped if its .json already exists under fold_results_dir.
# =============================================================================
set -e

# Repo root is this script's own parent directory (src/train_gpt.py lives
# alongside it), resolved from the script's own location so this works
# regardless of who clones it or where from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dataset=stress
dataset_path="./Stress_data"
matrix_p_path="./tMatrix_22x30_stress.npy"
gpu_id=0
pos_weight=-1.0  # 二分类，无 imbalanced 加权

# 数据预处理参数（沿用 author config 的 5秒/2chunk 切法，chunk_len 按新数据的
# 250Hz 重新换算：625*2=1250 samples = 5s @ 250Hz，每个 chunk 2.5s，不重叠）
num_chunks=2
chunk_len=625
chunk_ovlp=0

# 模型参数（LOSO 专用超参：lr=3e-4, wd=1e-1, bs=64；cls_head_layer=1ly，不是 3ly）
cls_head_layer='1ly'
lr=3e-4
wd=1e-1
batch_size=64

model_name=neurogpt
task_name=stress

exp_name="LOSO_full_1ly_lr${lr}_wd${wd}_bs${batch_size}"
fold_results_dir="./fold_results_neurogpt_stress"

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
    --num-decoding-classes=2 \
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
echo "Aggregate with: python3 aggregate_loso_results_stress.py --fold_results_dir ${fold_results_dir} --task ${task_name} --model ${model_name}"
