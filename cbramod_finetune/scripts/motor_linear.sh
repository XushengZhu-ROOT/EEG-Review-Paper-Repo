dataset=MotorTask
dataset_path=./AllSubjects_Epochs
exp_name=exp_author_config
gpu_id=0
chan_size=20
window_size=1
epochs=50
lr=0.0001
wd=0.00002
bs=64
classifier=all_patch_reps 
seed=0

freeze_type=linear_probe

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--frozen"
else
    FREEZE_ARG=""
fi

python finetune_main.py \
    --seed ${seed} \
    --epochs ${epochs} \
    --downstream_dataset ${dataset} \
    --datasets_dir ${dataset_path} \
    --optimizer  AdamW \
    --num_of_classes 6 \
    --channel_size ${chan_size} \
    --window_size ${window_size} \
    ${FREEZE_ARG} \
    --classifier ${classifier} \
    --use_pretrained_weights True \
    --foundation_dir ./pretrained_weights/pretrained_weights.pth \
    --cuda ${gpu_id} \
    --model_dir ./models_weights/${dataset}/${exp_name}-${classifier}-${freeze_type} \
    --batch_size ${bs} \
    --lr ${lr} \
    --weight_decay ${wd}

