#!/bin/bash
set -e
cd /home/dung/Documents/EEG-Review-Paper-Repo/biot_finetune


#--------- KaggleERN - fintune all (全微調) ---------#
#---------           - author config (一組) -------------#
#---------           - 3 層 Classifier -------------#

### 1. 作者設置, 全通道，全部微調，3層
dataset=ISRUC
# dataset_path=寫在run_binary_supervised.py裡
output_dir=${dataset}-posWeight${pos_weight}-rerun

gpu_id=0
dataset_channels=6
sample_length=5
epochs=50
pretrain_model_channels=16
pretrain_path=pretrained-models/EEG-PREST-${pretrain_model_channels}-channels.ckpt
classifier_type=CBraMod_3lyStyle_LayerNorm-BIOT # 3層
layers=3ly
n_classes=5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# lr=0.001
# wd=0.00001
# bs=512
# pos_weight=0.413

# freeze_type=all

# if [ "${freeze_type}" = "linear_probe" ]; then
#     FREEZE_ARG="--freeze_backbone"
# else
#     FREEZE_ARG=""
# fi

# exp_name=exp_author_config-${classifier_type}_${layers}-${freeze_type}

# CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
#     --exp_name ${exp_name} \
#     --dataset ${dataset} \
#     --n_classes ${n_classes} \
#     --dataset_channels ${dataset_channels} \
#     --in_channels ${pretrain_model_channels} \
#     --sampling_rate 200 \
#     --token_size 200 \
#     --hop_length 100 \
#     --sample_length ${sample_length} \
#     --batch_size ${bs} \
#     --lr ${lr} \
#     --weight_decay ${wd} \
#     --pos_weight ${pos_weight} \
#     --epochs ${epochs} \
#     --model ${classifier_type} \
#     --pretrain_model_path ${pretrain_path} \
#     --output_dir ${output_dir} \
#     ${FREEZE_ARG} 


#--------- KaggleERN - fintune all (全微調) ---------#
#---------           - hpo -------------#
#---------           - 3 層 Classifier -------------#

### 2. 作者設置, 全通道，全部微調，3層 + 超參數搜尋

BS_LIST=(256 512 1024)
LR_LIST=(0.005 0.001 0.0005)
WD_LIST=(0.001 0.00001)

count=0
for bs in "${BS_LIST[@]}"; do
  # 遍歷 Learning Rate
  for lr in "${LR_LIST[@]}"; do
    # 遍歷 Weight Decay
    for wd in "${WD_LIST[@]}"; do
        count=$((count + 1))
        exp_name=exp_hpo${count}-${classifier_type}_${layers}-${freeze_type}
        # 顯示目前正在執行的組合
        echo "================================================================"
        echo "Running experiment: ${exp_name}"
        echo "Batch Size (bs): ${bs}"
        echo "Learning Rate (lr): ${lr}"
        echo "Weight Decay (wd): ${wd}"
        echo "================================================================"

        CUDA_VISIBLE_DEVICES=${gpu_id} python3 run_multiclass_supervised.py \
            --exp_name ${exp_name} \
            --dataset ${dataset} \
            --n_classes ${n_classes} \
            --dataset_channels ${dataset_channels} \
            --in_channels ${pretrain_model_channels} \
            --sampling_rate 200 \
            --token_size 200 \
            --hop_length 100 \
            --sample_length ${sample_length} \
            --batch_size ${bs} \
            --lr ${lr} \
            --weight_decay ${wd} \
            --epochs ${epochs} \
            --model ${classifier_type} \
            --pretrain_model_path ${pretrain_path} \
            --output_dir ${output_dir} \
            ${FREEZE_ARG} 
        echo "--- Finished experiment: ${exp_name} ---"
        echo "" 
    done
  done
done


# #--------- KaggleERN - linear probe (只訓練頭) ---------#
# #---------           - author config (一組) -------------#
# #---------           - 3 層 Classifier -------------#

# ### 3. 作者設置, 全通道，只訓練頭，3層
# dataset=KaggleERN
# # dataset_path=寫在run_binary_supervised.py裡

# gpu_id=2
# dataset_channels=56
# sample_length=3
# epochs=50
# pretrain_model_channels=16
# pretrain_path=pretrained-models/EEG-PREST-${pretrain_model_channels}-channels.ckpt
# classifier_type=CBraMod_3lyStyle_LayerNorm-BIOT # 3層
# layers=3ly

# lr=0.001
# wd=0.00001
# bs=512
# pos_weight=0.413

# freeze_type=linear_probe

# if [ "${freeze_type}" = "linear_probe" ]; then
#     FREEZE_ARG="--freeze_backbone"
# else
#     FREEZE_ARG=""
# fi

# exp_name=exp_author_config-${classifier_type}_${layers}-${freeze_type}

# CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
#     --exp_name ${exp_name} \
#     --dataset ${dataset} \
#     --n_classes ${n_classes} \
#     --dataset_channels ${dataset_channels} \
#     --in_channels ${pretrain_model_channels} \
#     --sampling_rate 200 \
#     --token_size 200 \
#     --hop_length 100 \
#     --sample_length ${sample_length} \
#     --batch_size ${bs} \
#     --lr ${lr} \
#     --weight_decay ${wd} \
#     --pos_weight ${pos_weight} \
#     --epochs ${epochs} \
#     --model ${classifier_type} \
#     --pretrain_model_path ${pretrain_path} \
#     --output_dir ${dataset}-posWeight${pos_weight}\
#     ${FREEZE_ARG} 