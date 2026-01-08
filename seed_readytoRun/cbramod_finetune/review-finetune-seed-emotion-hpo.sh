

BS_LIST=(64 256 512)
LR_LIST=(0.0005 0.0001 0.00005)
WD_LIST=(0.002 0.00002)

dataset=SEED-Emotion
dataset_path=./seed_data
gpu_id=0
chan_size=62
window_size=4
epochs=50
classifier=all_patch_reps 
seed=62

freeze_type=all

if [ "${freeze_type}" = "linear_probe" ]; then
    FREEZE_ARG="--frozen"
else
    FREEZE_ARG=""
fi

exp_count=1
for bs in "${BS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for wd in "${WD_LIST[@]}"; do
        exp_name=hpo_exp${exp_count}_lr${lr}_wd${wd}_bs${bs}
        echo "--- [STARTING EXPERIMENT] ---"
        echo "BatchSize: ${bs}, LearningRate: ${lr}, WeightDecay: ${wd}"
        echo "Exp Name: ${exp_name}"
        echo "-------------------------------"
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
        echo "--- [EXPERIMENT #${exp_count} FINISHED] ---"
        echo "" 
        exp_count=$((exp_count + 1))
    done
  done
done

