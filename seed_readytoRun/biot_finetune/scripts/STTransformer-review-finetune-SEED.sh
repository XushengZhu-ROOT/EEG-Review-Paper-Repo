# set -e

dataset=SEED-ST
# dataset_path=寫在run_multiclass_supervised.py裡
gpu_id=0
sample_length=5
epochs=50
classifier_type=STTransformer
dataset_channels=62

output_dir=${dataset}-${classifier_type}


#--------- SEED-ST - STTransformer (全微調) ---------#
### hpo search

# 作者配置
# lr=0.001
# wd=0.00001
# bs=512

BS_LIST=(256 512 1024)
LR_LIST=(0.005 0.001 0.0005)
WD_LIST=(0.001 0.00001)

for bs in "${BS_LIST[@]}"; do
    # 遍歷 Learning Rate (lr) 列表
    for lr in "${LR_LIST[@]}"; do
        # 遍歷 Weight Decay (wd) 列表
        for wd in "${WD_LIST[@]}"; do
            
            exp_name=exp_hpo-lr${lr}-wd${wd}-bs${bs}
            
            echo "--------------------------------------------------------"
            echo "➡️ 執行實驗: ${exp_name}"
            echo "   Batch Size: ${bs}, Learning Rate: ${lr}, Weight Decay: ${wd}"
            echo "--------------------------------------------------------"
            
            CUDA_VISIBLE_DEVICES=${gpu_id} python run_multiclass_supervised.py \
            --exp_name ${exp_name} \
            --dataset ${dataset} \
            --n_classes 7 \
            --in_channels ${dataset_channels} \
            --sampling_rate 250 \
            --sample_length ${sample_length} \
            --batch_size ${bs} \
            --lr ${lr} \
            --weight_decay ${wd} \
            --epochs ${epochs} \
            --model ${classifier_type} \
            --output_dir ${output_dir}
            # (可選) 在每次訓練結束後添加一個延遲，以確保 GPU 釋放資源
            sleep 5
        done
    done
done

echo "--- ✅ 所有實驗執行完畢 ✅ ---"

