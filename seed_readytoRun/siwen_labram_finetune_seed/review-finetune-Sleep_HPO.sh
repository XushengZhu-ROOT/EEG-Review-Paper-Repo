dataset=Sleep
dataset_path=./labram_sleep_data
chan_size=6
epochs=50
seed=0
gpu_id=0
pos_weight=1.0
exp_name=expauthor
freeze_type=all
model_type=3ly
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    FREEZE_ARG=""
fi
if [ "${model_type}" = "3ly" ]; then
    MODEL_ARG="labram_base_patch200_200_cbramod3lyclassifier"
else
    MODEL_ARG="labram_base_patch200_200"
fi

# 超参数搜索：Sleep数据集6通道，内存占用较小，可以使用较大的batch_size
# 保持有效batch_size=512（通过update_freq调整）
BS_LIST=(128 256 512)
LR_LIST=(0.0009 0.0005 0.00009)
WD_LIST=(0.05 0.005 0.0001)

exp_count=1
for bs in "${BS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for wd in "${WD_LIST[@]}"; do
        # 根据batch_size计算update_freq以保持有效batch_size=512
        if [ ${bs} -eq 128 ]; then
            update_freq=4
        elif [ ${bs} -eq 256 ]; then
            update_freq=2
        else
            update_freq=1
        fi
        
        exp_name=hpo_exp${exp_count}_lr${lr}_wd${wd}_bs${bs}

        exp_dir="./new_ckpt/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type}"
        if [ -d "${exp_dir}" ]; then
            echo "Skip ${exp_name}, directory exists: ${exp_dir}"
            exp_count=$((exp_count + 1))
            continue
        fi

        echo "--- [STARTING EXPERIMENT] ---"
        echo "Physical BatchSize: ${bs}, UpdateFreq: ${update_freq}, Effective BatchSize: $((bs * update_freq))"
        echo "LearningRate: ${lr}, WeightDecay: ${wd}"
        echo "Exp Name: ${exp_name}"
        echo "-------------------------------"
        CUDA_VISIBLE_DEVICES=${gpu_id} python run_class_finetuning.py \
            --output_dir ./new_ckpt/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
            --log_dir ./log/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type} \
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
            --update_freq ${update_freq} \
            --warmup_epochs 3 \
            --save_ckpt_freq 5 \
            --disable_rel_pos_bias \
            --disable_qkv_bias
        echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
        echo "" 

        exp_count=$((exp_count + 1))
    done
  done
done
