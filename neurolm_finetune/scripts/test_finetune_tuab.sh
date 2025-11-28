OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=5 train_instruction.py \
    --dataset_dir /work/HHRI-AI/YW/Yirong/NeuroLM/data/ \
    --out_dir checkpoints/FinetuneLEM-NeuroLM-TUAB-test \
    --NeuroLM_path checkpoints/NeuroLM-B.pt \
    --wandb_log \
    --wandb_project FinetuneLEM-NeuroLM-TUAB-test \
    --wandb_runname test \
    --wandb_api_key 88588f69f35ddd71a8eb1f079d05a5bf43171b95 \
    --eeg_batch_size 4 \
    --text_batch_size 4 