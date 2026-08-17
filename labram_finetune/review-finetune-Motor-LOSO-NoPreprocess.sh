#!/bin/bash
# =============================================================================
# Motor fine-tune launcher (20-fold subject-independent LOSO, val-selected epoch)
# -- NO_preprocess variant: same protocol as review-finetune-Motor-LOSO.sh,
#    just pointed at AllSubjects_Epochs_NO_preprocess instead of
#    AllSubjects_Epochs, with its own output/results dirs so it never
#    collides with the already-completed preprocessed run.
# =============================================================================
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=Motor
dataset_path=./AllSubjects_Epochs_NO_preprocess
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

exp_name=exp_author_config_LOSO_noPreprocess
parent_dir=./new_ckpt/${dataset}-posWeight${pos_weight}-LOSO-noPreprocess/${exp_name}-${model_type}-${freeze_type}
fold_results_dir=./fold_results_labram_noPreprocess

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
