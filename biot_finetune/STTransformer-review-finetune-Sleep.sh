# set -e

dataset=Sleep
# dataset_path=寫在run_multiclass_supervised.py裡
gpu_id=0
sample_length=30
epochs=50
classifier_type=STTransformer
dataset_channels=6

output_dir=${dataset}-${classifier_type}


#--------- Sleep-ST - STTransformer (全微調) ---------#
### hpo search

# 作者配置
# lr=0.001
# wd=0.00001
# bs=512

BS_LIST=(256 512 1024)
LR_LIST=(0.005 0.001 0.0005)
WD_LIST=(0.001 0.00001)

# 默认的token_size和hop_length（与run_multiclass_supervised.py中的默认值一致）
token_size=200
hop_length=100

for bs in "${BS_LIST[@]}"; do
    # 遍歷 Learning Rate (lr) 列表
    for lr in "${LR_LIST[@]}"; do
        # 遍歷 Weight Decay (wd) 列表
        for wd in "${WD_LIST[@]}"; do
            
            exp_name=exp_hpo-lr${lr}-wd${wd}-bs${bs}
            
            # 构建log目录路径（与run_multiclass_supervised.py中的version格式一致）
            # version = f"{dataset}-{model}-{lr}-{batch_size}-{sampling_rate}-{token_size}-{hop_length}"
            log_dir="log/${dataset}-${classifier_type}-${lr}-${bs}-250-${token_size}-${hop_length}"
            test_file="${log_dir}/test_confusion_matrix_epoch_${epochs}.npy"
            
            # 检查实验是否已完成
            if [ -f "$test_file" ]; then
                echo "--------------------------------------------------------"
                echo "⏭️  跳過已完成實驗: ${exp_name}"
                echo "   測試文件已存在: ${test_file}"
                echo "   Batch Size: ${bs}, Learning Rate: ${lr}, Weight Decay: ${wd}"
                echo "--------------------------------------------------------"
                continue
            fi
            
            echo "--------------------------------------------------------"
            echo "➡️ 執行實驗: ${exp_name}"
            echo "   Batch Size: ${bs}, Learning Rate: ${lr}, Weight Decay: ${wd}"
            echo "--------------------------------------------------------"
            
            CUDA_VISIBLE_DEVICES=${gpu_id} python run_multiclass_supervised.py \
            --exp_name ${exp_name} \
            --dataset ${dataset} \
            --n_classes 5 \
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
