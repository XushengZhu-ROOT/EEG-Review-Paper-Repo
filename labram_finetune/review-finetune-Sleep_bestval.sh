dataset=Sleep
dataset_path=./labram_sleep_data
chan_size=6
epochs=50
seed=0
lr=9e-4
wd=5e-2
bs=512
update_freq=1     # 按 bs=512 时 HPO 网格里的公式（保持有效 batch_size=512）
gpu_id=0
pos_weight=1.0
exp_name=bestval_lr${lr}_wd${wd}_bs${bs}
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
CUDA_VISIBLE_DEVICES=${gpu_id} python run_class_finetuning.py \
     --output_dir ./new_ckpt/${dataset}-posWeight${pos_weight}-swien_config/${exp_name}-${model_type}-${freeze_type}\
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
     --disable_qkv_bias \
     --task_name sleep \
     --fold_results_dir ./fold_results_labram_sleep

python compute_metrics_from_npz.py --npz_dir ./fold_results_labram_sleep --task sleep --model labram --n_classes 5 --ci --n_bootstrap 1000
