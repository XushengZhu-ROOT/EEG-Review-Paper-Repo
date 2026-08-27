#!/bin/bash
# [KaggleERN bestval][STTransformer] 用 STTransformer 专属的 250Hz KaggleERN 数据
# （跟 Sleep/Motion/Stress 的 -ST 变体同一套模式），不跟 BIOT 共用 s42_n56-biot。
#
# 下面这组超参数是结果统计/kaggle.csv 里 STTransformer 用旧共享数据调出来的最优组合，
# 仅作起点；换了数据后的结果不能跟 kaggle.csv 里旧的 66.23% 直接比较。
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

# --sampling_rate 填 200（不是 250）：数据已经离线 resample 到 250Hz，KaggleERNLoader
# 只有 sampling_rate!=200 才会二次 resample，这里填 200 是让它保持原样，不代表数据是 200Hz。
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
