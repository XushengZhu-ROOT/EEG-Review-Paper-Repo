set -e
n_gpu=4
gpu_id=2,3,6,7

# ---- 固定參數 ----
dataset_dir=/work/HHRI-AI/UCSD_EEG/eeg_data/EEG_data/motor-neurolm-for_reviwer_comment          # 內含 train/val/test，會被忽略原始劃分、按被試重新分折
test_data_dir=/work/HHRI-AI/YW/Yirong/NeuroLM/data/text
NeuroLM_path=/work/HHRI-AI/YW/Yirong/NeuroLM/checkpoints/NeuroLM-B.pt

dataset=MOTOR
model_name=NeuroLM                         # 影響輸出檔名 {task}_{model}_fold{i}.npz/json
task_name=motor                            # 輸出檔名的 task 前綴（預設就是小寫資料集名）

# ---- LOSO 設定 ----
N_FOLDS=20                                  # motor 被試數；跑 0..N_FOLDS-1 折

# ---- 訓練超參（LOSO 沿用「這一組」，通常由前面 HPO 選出後固定）----
epochs=5
lr=5e-4
wd=0.1
min_lr=5e-5
adamw_b1=0.9
adamw_b2=0.95
tbs=9                                      # text_batch_size
ibs=36                                      # eeg_batch_size

# 所有折的結果統一存到這個資料夾，方便事後彙總
results_dir=results/${dataset}/loso_lr${lr}_wd${wd}

for (( fold=0; fold<N_FOLDS; fold++ )); do
    exp_name="loso_fold$(printf '%02d' ${fold})_lr${lr}_wd${wd}"
    out_dir=checkpoints/${dataset}/${exp_name}

    echo "--- [STARTING FOLD #${fold}] ---"
    echo "Exp Name: ${exp_name}"
    echo "Out dir: ${out_dir} | Results dir: ${results_dir}"
    echo "LR: ${lr}, WD: ${wd}, Text BS: ${tbs}, EEG BS: ${ibs}"
    echo "-----------------------------------"

    CUDA_VISIBLE_DEVICES=${gpu_id} MP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=${n_gpu} train_instruction.py \
    --dataset_dir ${dataset_dir} \
    --text_data_dir ${test_data_dir} \
    --out_dir ${out_dir} \
    --results_dir ${results_dir} \
    --NeuroLM_path ${NeuroLM_path} \
    --model_name ${model_name} \
    --task_name ${task_name} \
    --fold ${fold} \
    --n_folds ${N_FOLDS} \
    --wandb_log \
    --wandb_project Finetune-NeuroLM-${dataset}-LOSO \
    --wandb_runname ${exp_name} \
    --wandb_api_key 88588f69f35ddd71a8eb1f079d05a5bf43171b95 \
    --eeg_batch_size ${ibs} \
    --text_batch_size ${tbs} \
    --epochs ${epochs} \
    --learning_rate ${lr} \
    --min_lr ${min_lr} \
    --beta1 ${adamw_b1} \
    --beta2 ${adamw_b2} \
    --weight_decay ${wd}

    echo "--- [FOLD #${fold} FINISHED] ---"
    echo ""
done

echo "========================================"
echo "All ${N_FOLDS} LOSO folds completed."
echo "Results saved under: ${results_dir}"
echo "  ${task_name}_${model_name}_fold{00..$(printf '%02d' $((N_FOLDS-1)))}.npz / .json"
echo "========================================"