# set -e

dataset_channels=30
dataset=CustomStress-${dataset_channels}chan
# dataset_path=寫在run_binary_supervised.py裡
gpu_id=2
sample_length=5
epochs=50
pretrain_model_channels=16
pretrain_path=pretrained-models/EEG-PREST-${pretrain_model_channels}-channels.ckpt
classifier_type=LabramClassifier-BIOT
layers=1ly

#--------- STRESS - finetune (全微調) ---------#
#---------           - author config (一組) -------------#
#---------           - 1 層 Classifier -------------#
### 1. 作者設置, 全通道，全部微調，1層

# lr, wd, bs 是作者配置
lr=0.001
wd=0.00001
bs=512

# 全微調
freeze_type=all
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    # 如果不是，我們就傳入一個空字串，等於什麼都不加
    FREEZE_ARG=""
fi

exp_name=exp_author_config-${classifier_type}_${layers}-${freeze_type}
CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
    --exp_name ${exp_name} \
    --dataset ${dataset} \
    --n_classes 1 \
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
    --output_dir ${dataset}-${classifier_type}\
    ${FREEZE_ARG} 


#--------- STRESS - finetune (全微調) ---------#
#---------           - hpo -------------#
#---------           - 1 層 Classifier -------------#
### 2. 作者設置, 全通道，全部微調，1層 +  超參數搜尋

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

        CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
            --exp_name ${exp_name} \
            --dataset ${dataset} \
            --n_classes 1 \
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
            --output_dir ${dataset}-${classifier_type}\
            ${FREEZE_ARG} 
        echo "--- Finished experiment: ${exp_name} ---"
        echo "" 
    done
  done
done


#--------- STRESS - Linear Probe (只訓練頭) ---------#
#---------           - author config (一組) -------------#
#---------           - 1 層 Classifier -------------#
### 3. 作者設置, 全通道，只訓練頭，1層

# lr, wd, bs 是作者配置
lr=0.001
wd=0.00001
bs=512

# 只訓練頭
freeze_type=linear_probe
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    # 如果不是，我們就傳入一個空字串，等於什麼都不加
    FREEZE_ARG=""
fi

exp_name=exp_author_config-${classifier_type}_${layers}-${freeze_type}
CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
    --exp_name ${exp_name} \
    --dataset ${dataset} \
    --n_classes 1 \
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
    --output_dir ${dataset}-${classifier_type}\
    ${FREEZE_ARG} 

    
#--------- STRESS - Linear Probe (只訓練頭) ---------#
#---------           - hpo -------------#
#---------           - 1 層 Classifier -------------#
### 4. 作者設置, 全通道，只訓練頭，1層 + 超參數搜尋

BS_LIST=(256 512 1024)
LR_LIST=(0.005 0.001 0.0005)
WD_LIST=(0.001 0.00001)

# 只訓練頭
freeze_type=linear_probe
if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--freeze_backbone"
else
    # 如果不是，我們就傳入一個空字串，等於什麼都不加
    FREEZE_ARG=""
fi

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

        CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
            --exp_name ${exp_name} \
            --dataset ${dataset} \
            --n_classes 1 \
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
            --output_dir ${dataset}-${classifier_type}\
            ${FREEZE_ARG} 
        echo "--- Finished experiment: ${exp_name} ---"
        echo "" 
    done
  done
done 