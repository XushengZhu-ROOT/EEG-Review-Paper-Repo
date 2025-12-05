set -e
dataset=stress
dataset_path="/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42"
matrix_p_path="../inputs/tMatrix_22x30_stress.npy"
gpu_id=2
pos_weight=-1.0 # 無 imbalanced

num_chunks=2
# window_size=2
chunk_len=500 # 5 秒鐘切成 2 個chunk (sample rate=200)
chunk_ovlp=0 # 2個 chunk 不重疊

#--------- Stress - NeuroGPT (全微調 + hpo) ---------#
cls_head_layer='1ly'
# 作者 config
# batch_size=32
# lr=0.0001
# wd=0.1

LR_LIST=(3e-4 1e-4 6e-5)
WD_LIST=(0.0 0.1)
BS_LIST=(16 32 64)

exp_count=1

for batch_size in "${BS_LIST[@]}"; do
    
    # 2. 學習率 (Learning Rate) 迴圈
    for lr in "${LR_LIST[@]}"; do
        
        # 3. 權重衰減 (Weight Decay) 迴圈
        for wd in "${WD_LIST[@]}"; do
            
            # 命名本次實驗，包含所有 HPO 參數
            exp_name="hpo_exp${exp_count}_${cls_head_layer}_lr${lr}_wd${wd}_bs${batch_size}"
            
            echo "--- 🏃 執行實驗 #${exp_count}: ${exp_name} ---"
            echo "LR: ${lr}, WD: ${wd}, Batch Size: ${batch_size}"
            
            CUDA_VISIBLE_DEVICES=${gpu_id} python3 ../src/train_gpt.py \
            --training-style='decoding' \
            --num-decoding-classes=2 \
            --training-steps=10000  \
            --eval_every_n_steps=500 \
            --log-every-n-steps=500 \
            --num_chunks=${num_chunks} \
            --per-device-training-batch-size=${batch_size} \
            --per-device-validation-batch-size=${batch_size} \
            --chunk_len=${chunk_len} \
            --chunk_ovlp=${chunk_ovlp} \
            --run-name=${exp_name} \
            --ft-only-encoder='True' \
            --fold_i=0 \
            --num-encoder-layers=6 \
            --num-hidden-layers=6 \
            --learning-rate=${lr} \
            --weight-decay=${wd} \
            --use-encoder='True' \
            --embedding-dim=1024  \
            --pretrained-model='../pretrained_model/pytorch_model.bin' \
            --dataset-name=${dataset} \
            --dst-data-path=${dataset_path} \
            --matrix_p_path=${matrix_p_path} \
            --seed=0 \
            --pos_weight=${pos_weight} \
            --log-dir="../results/${dataset}" \
            --metric_for_best_model="eval_validation_bacc" \
            --cls_head_layer=${cls_head_layer}
            echo "--- 實驗 #${exp_count} 完成 ---"
            echo ""
            exp_count=$((exp_count + 1))
        done
    done
done

echo "🎉 所有超參數組合實驗已完成！共 $((exp_count - 1)) 個實驗。"


#--------- Stress - NeuroGPT (只訓練頭 + hpo) ---------#
dataset=stress
dataset_path="/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42"
matrix_p_path="../inputs/tMatrix_22x30_stress.npy"
gpu_id=2
pos_weight=-1.0 # 無 imbalanced

num_chunks=2
# window_size=2
chunk_len=500 # 5 秒鐘切成 2 個chunk (sample rate=200)
chunk_ovlp=0 # 2個 chunk 不重疊

cls_head_layer='1ly'

# 作者設置
# batch_size=32
# lr=0.0001
# wd=0.1

LR_LIST=(3e-4 1e-4 6e-5)
WD_LIST=(0.0 0.1)
BS_LIST=(16 32 64)

exp_count=1

for batch_size in "${BS_LIST[@]}"; do
    
    # 2. 學習率 (Learning Rate) 迴圈
    for lr in "${LR_LIST[@]}"; do
        # 3. 權重衰減 (Weight Decay) 迴圈
        for wd in "${WD_LIST[@]}"; do
            # 命名本次實驗，包含所有 HPO 參數
            exp_name="hpo_exp${exp_count}_${cls_head_layer}_lr${lr}_wd${wd}_bs${batch_size}"
            
            echo "--- 🏃 執行實驗 #${exp_count}: ${exp_name} ---"
            echo "LR: ${lr}, WD: ${wd}, Batch Size: ${batch_size}"
            
            CUDA_VISIBLE_DEVICES=${gpu_id} python3 ../src/train_gpt.py \
            --training-style='decoding' \
            --num-decoding-classes=2 \
            --training-steps=10000  \
            --eval_every_n_steps=500 \
            --log-every-n-steps=500 \
            --num_chunks=${num_chunks} \
            --per-device-training-batch-size=${batch_size} \
            --per-device-validation-batch-size=${batch_size} \
            --chunk_len=${chunk_len} \
            --chunk_ovlp=${chunk_ovlp} \
            --run-name=${exp_name} \
            --ft-only-encoder='True' \
            --fold_i=0 \
            --num-encoder-layers=6 \
            --num-hidden-layers=6 \
            --learning-rate=${lr} \
            --weight-decay=${wd} \
            --use-encoder='True' \
            --embedding-dim=1024  \
            --pretrained-model='../pretrained_model/pytorch_model.bin' \
            --dataset-name=${dataset} \
            --dst-data-path=${dataset_path} \
            --matrix_p_path=${matrix_p_path} \
            --seed=0 \
            --pos_weight=${pos_weight} \
            --log-dir="../results/${dataset}-freeze_encoder" \
            --metric_for_best_model="eval_validation_bacc" \
            --cls_head_layer=${cls_head_layer} \
            --freeze-encoder='True'
            echo "--- 實驗 #${exp_count} 完成 ---"
            echo ""
            exp_count=$((exp_count + 1))
        done
    done
done

echo "🎉 所有超參數組合實驗已完成！共 $((exp_count - 1)) 個實驗。"