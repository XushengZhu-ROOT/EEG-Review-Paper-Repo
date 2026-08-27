#!/bin/bash
# [KaggleERN bestval][STTransformer] 用结果统计/kaggle.csv 里 STTransformer 一行
# best val bacc 最高的超参数组合重跑一遍（lr=1e-3, wd=1e-5, bs=256,
# val_bacc=66.23%, best_epoch=48）。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=KaggleERN
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
    --pos_weight ${pos_weight} \
    --epochs ${epochs} \
    --model ${classifier_type} \
    --output_dir ${output_dir} \
    --task_name kaggleern \
    --model_name st \
    --fold_results_dir ./fold_results_st_kaggleern

python compute_metrics_from_npz.py --npz_dir ./fold_results_st_kaggleern --task kaggleern --model st --n_classes 2 --ci --n_bootstrap 1000
cat ./fold_results_st_kaggleern/kaggleern_st.json
