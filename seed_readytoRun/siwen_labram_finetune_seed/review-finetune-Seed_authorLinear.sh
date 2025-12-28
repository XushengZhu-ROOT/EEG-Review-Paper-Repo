dataset=Seed
dataset_path=./seed_data
chan_size=62
epochs=50
seed=0
lr=5e-4
wd=0.05
bs=512
gpu_id=0
pos_weight=1.0
exp_name=expauthor
freeze_type=linear_probe
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
     --update_freq 8 \
     --warmup_epochs 3 \
     --save_ckpt_freq 5 \
     --disable_rel_pos_bias \
     --disable_qkv_bias

