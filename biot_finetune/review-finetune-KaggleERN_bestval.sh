#!/bin/bash
# [KaggleERN bestval][BIOT] 用结果统计/kaggle.csv 里 BIOT 一行 best val bacc 最高的
# 超参数组合重跑一遍（lr=5e-4, wd=1e-5, bs=256, val_bacc=61.79%, best_epoch=33），
# 训练时用 val_bacc 最好的那个 epoch 存 kaggleern_biot_val.npz/_test.npz/
# kaggleern_biot.json，不用重新去翻训练日志。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=KaggleERN
gpu_id=0
dataset_channels=56
sample_length=3
epochs=50
pretrain_model_channels=16
pretrain_path=pretrained-models/EEG-PREST-${pretrain_model_channels}-channels.ckpt
classifier_type=CBraMod_3lyStyle_LayerNorm-BIOT
pos_weight=0.413

lr=5e-4
wd=1e-5
bs=256

exp_name=bestval-${classifier_type}-lr${lr}-wd${wd}-bs${bs}
output_dir=${dataset}-${classifier_type}-bestval

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
    --pos_weight ${pos_weight} \
    --epochs ${epochs} \
    --model ${classifier_type} \
    --pretrain_model_path ${pretrain_path} \
    --output_dir ${output_dir} \
    --task_name kaggleern \
    --model_name biot \
    --fold_results_dir ./fold_results_biot_kaggleern

python compute_metrics_from_npz.py --npz_dir ./fold_results_biot_kaggleern --task kaggleern --model biot --n_classes 2 --ci --n_bootstrap 1000
cat ./fold_results_biot_kaggleern/kaggleern_biot.json
