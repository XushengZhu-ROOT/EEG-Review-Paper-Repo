#!/bin/bash
# [KaggleERN bestval][STTransformer] 之前这个脚本跟 BIOT 共用同一份 s42_n56-biot
# 数据（200Hz，无 STTransformer 专属滤波），跟 Sleep/Motion/Stress 里 STTransformer
# 都有自己独立预处理数据（250Hz + 4-40Hz带通+notch，见
# preprocessing/preprocess_KaggleERN_new.ipynb 的 extract_epochs_sttransformer）
# 的模式不一致，已经改成读独立的 KaggleERN-ST 数据。
#
# !!! dataset_dir 换成你服务器上真实的 STTransformer 专属 KaggleERN 数据路径 !!!
# （train/val/test 三个子目录，pickle 里 'signal'/'label' 字段，跟 BIOT 那份格式一致，
# 只是预处理参数不同）
#
# 下面这组超参数（lr=1e-3, wd=1e-5, bs=256）是结果统计/kaggle.csv 里 STTransformer
# 用旧的共享 BIOT 数据调出来的最优组合，仅作为这次换数据后的起点，重新用这份
# STTransformer 专属数据跑出来的 val/test bacc 不能跟 kaggle.csv 里旧的 66.23% 直接
# 比较——这是预期之内的，算是一次新实验。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=KaggleERN-ST
dataset_dir="kaggle_data_st"
gpu_id=0
dataset_channels=56
sample_length=3
epochs=50
classifier_type=STTransformer
pos_weight=0.413

lr=1e-3
wd=1e-5
bs=256

exp_name=bestval-${classifier_type}-lr${lr}-wd${wd}-bs${bs}
output_dir=${dataset}-${classifier_type}-bestval

# --sampling_rate 下面故意还是填 200（不是 250）：KaggleERNLoader 只有
# sampling_rate!=default_rate(200) 时才会二次 resample，而它内部按 10*sampling_rate
# 算目标长度（是给别的 10s 任务用的公式，对 3s 的 KaggleERN 不成立）。数据已经在
# 预处理阶段被离线 resample 到 250Hz 了，这里填 200 只是告诉 loader "不要再动它"，
# 不代表数据真的是 200Hz。
CUDA_VISIBLE_DEVICES=${gpu_id} python run_binary_supervised.py \
    --exp_name ${exp_name} \
    --dataset ${dataset} \
    --dataset_dir ${dataset_dir} \
    --n_classes 1 \
    --in_channels ${dataset_channels} \
    --sampling_rate 200 \
    --sample_length ${sample_length} \
    --batch_size ${bs} \
    --lr ${lr} \
    --weight_decay ${wd} \
    --pos_weight ${pos_weight} \
    --epochs ${epochs} \
    --model ${classifier_type} \
    --output_dir ${output_dir} \
    --task_name kaggleern \
    --model_name st \
    --fold_results_dir ./fold_results_st_kaggleern

python compute_metrics_from_npz.py --npz_dir ./fold_results_st_kaggleern --task kaggleern --model st --n_classes 2 --ci --n_bootstrap 1000
cat ./fold_results_st_kaggleern/kaggleern_st.json
