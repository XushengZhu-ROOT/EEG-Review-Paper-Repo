set -e
n_gpu=2
gpu_id=2,3
# 固定參數
dataset_dir=seed_data
test_data_dir=text
NeuroLM_path=checkpoints/NeuroLM-B.pt

dataset=SEED7
exp_name=exp_author_config
out_dir=checkpoints/${dataset}/${exp_name}

chan_size=62
epochs=5
min_lr=5e-5
adamw_b1=0.9
adamw_b2=0.95

# bs=[(7, 28), (14, 42)]
# lr=[ , 5e-4, ]
# wd=[0.1, 0.001]
# text_batch_size=14 # 20*4=80
# instruction_batch_size=42 #90*4=360
# learning_rate=5e-4
# weight_decay=0.1

LR_LIST=(5e-6 1e-5 5e-5 5e-4)
WD_LIST=(0.0 0.01 0.1)
TEXT_BS_LIST=(13 12 9) # 13+52是3張剩下60G的卡的極限
INST_BS_LIST=(52 48 36)

exp_count=1
for i in "${!TEXT_BS_LIST[@]}"; do
    tbs=${TEXT_BS_LIST[$i]}
    ibs=${INST_BS_LIST[$i]}
    
    for lr in "${LR_LIST[@]}"; do
        
        for wd in "${WD_LIST[@]}"; do
            
            exp_name="hpo_exp${exp_count}_lr${lr}_wd${wd}_textbs${tbs}_instbs${ibs}"
            
            out_dir=checkpoints/${dataset}/${exp_name}
            
            echo "--- [STARTING EXPERIMENT #${exp_count}] ---"
            echo "Exp Name: ${exp_name}"
            echo "Output dir: ${out_dir}"
            echo "LR: ${lr}, WD: ${wd}, Text BS: ${tbs}, EEG BS: ${ibs}"
            echo "-----------------------------------"
            
            CUDA_VISIBLE_DEVICES=${gpu_id} MP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=${n_gpu} train_instruction.py \
            --dataset_dir ${dataset_dir} \
            --text_data_dir ${test_data_dir} \
            --out_dir ${out_dir} \
            --NeuroLM_path ${NeuroLM_path} \
            --wandb_log \
            --wandb_project Finetune-NeuroLM-${dataset} \
            --wandb_runname ${exp_name} \
            --wandb_api_key 88588f69f35ddd71a8eb1f079d05a5bf43171b95 \
            --chan_size ${chan_size} \
            --eeg_batch_size ${ibs} \
            --text_batch_size ${tbs} \
            --epochs ${epochs} \
            --learning_rate ${lr} \
            --min_lr ${min_lr} \
            --beta1 ${adamw_b1} \
            --beta2 ${adamw_b2} \
            --weight_decay ${wd}
            
            echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
            echo ""
            
            exp_count=$((exp_count + 1))
            
        done # 結束 wd 迴圈
    done # 結束 lr 迴圈
done # 結束 bs 迴圈

echo "========================================"
echo "All HPO experiments completed."
echo "Total experiments run: $((exp_count - 1))"
echo "========================================"

