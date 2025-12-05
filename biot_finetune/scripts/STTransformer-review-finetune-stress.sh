# set -e

dataset_channels=30
dataset=CustomStress-${dataset_channels}chan
# dataset_path=寫在run_binary_supervised.py裡
gpu_id=2
sample_length=5
epochs=50
classifier_type=STTransformer


#--------- STRESS - STTransformer (全微調) ---------#
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
            
            CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
            --exp_name ${exp_name} \
            --dataset ${dataset} \
            --n_classes 1 \
            --in_channels ${dataset_channels} \
            --sampling_rate 200 \
            --sample_length ${sample_length} \
            --batch_size ${bs} \
            --lr ${lr} \
            --weight_decay ${wd} \
            --epochs ${epochs} \
            --model ${classifier_type} \
            --output_dir ${dataset}-${classifier_type}-rerun
            # (可選) 在每次訓練結束後添加一個延遲，以確保 GPU 釋放資源
            sleep 5
        done
    done
done

echo "--- ✅ 所有實驗執行完畢 ✅ ---"

# -------- 作者設置? -------------
# dataset_channels=30
# dataset=CustomStress-${dataset_channels}chan
# gpu_id=2
# sample_length=5
# epochs=50
# classifier_type=STTransformer

# lr=0.0005
# wd=0.00001
# bs=256

# exp_name=exp_hpo
# CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
# --exp_name ${exp_name} \
# --dataset ${dataset} \
# --n_classes 1 \
# --in_channels ${dataset_channels} \
# --sampling_rate 200 \
# --sample_length ${sample_length} \
# --batch_size ${bs} \
# --lr ${lr} \
# --weight_decay ${wd} \
# --epochs ${epochs} \
# --model ${classifier_type} \
# --output_dir ${dataset}-${classifier_type}-test