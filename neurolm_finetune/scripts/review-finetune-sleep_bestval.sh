set -e
n_gpu=3
gpu_id=1,2,3
# 固定參數
dataset_dir=sleep_data
test_data_dir=text
NeuroLM_path=checkpoints/NeuroLM-B.pt

dataset=SLEEP
exp_name=bestval_lr5e-05_wd0.0_textbs9_instbs36
out_dir=checkpoints/${dataset}/${exp_name}

epochs=5
min_lr=5e-5
adamw_b1=0.9
adamw_b2=0.95

lr=5e-5
wd=0.0
tbs=9
ibs=36

CUDA_VISIBLE_DEVICES=${gpu_id} MP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=${n_gpu} train_instruction.py \
    --dataset_dir ${dataset_dir} \
    --text_data_dir ${test_data_dir} \
    --out_dir ${out_dir} \
    --NeuroLM_path ${NeuroLM_path} \
    --eeg_batch_size ${ibs} \
    --text_batch_size ${tbs} \
    --epochs ${epochs} \
    --learning_rate ${lr} \
    --min_lr ${min_lr} \
    --beta1 ${adamw_b1} \
    --beta2 ${adamw_b2} \
    --weight_decay ${wd} \
    --save_ckpt \
    --model_name neurolm \
    --task_name sleep \
    --results_dir ./fold_results_neurolm_sleep

    # 如需 wandb 记录，照 review-finetune-sleep.sh 加：
    # --wandb_log --wandb_project Finetune-NeuroLM-${dataset} --wandb_runname ${exp_name} --wandb_api_key <your_key>

python compute_metrics_from_npz.py --npz_dir ./fold_results_neurolm_sleep --task sleep --model neurolm --n_classes 5 --ci --n_bootstrap 1000
cat ./fold_results_neurolm_sleep/sleep_neurolm.json

echo "========================================"
echo "Best-val rerun completed."
echo "========================================"
