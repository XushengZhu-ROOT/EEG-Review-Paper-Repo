dataset=Sleep
gpu_id=0
sample_length=30
epochs=50
classifier_type=STTransformer
dataset_channels=6

lr=1e-3
wd=1e-3
bs=256

exp_name=bestval-lr${lr}-wd${wd}-bs${bs}
output_dir=${dataset}-${classifier_type}

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
    --output_dir ${output_dir} \
    --task_name sleep \
    --model_name st \
    --fold_results_dir ./fold_results_st_sleep

python compute_metrics_from_npz.py --npz_dir ./fold_results_st_sleep --task sleep --model st --n_classes 5 --ci --n_bootstrap 1000
