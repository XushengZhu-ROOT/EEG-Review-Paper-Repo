python finetune_main.py \
--downstream_dataset CustomStress \
--datasets_dir /work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42 \
--num_of_classes 2 \
--model_dir /work/HHRI-AI/YW/Yirong/CBraMod/models_weights/finetune-Stress_noleak_30chan_no400up_seed_siwen42-4review-default-all_reps

python finetune_main.py \
--downstream_dataset CustomStress \
--datasets_dir /work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42 \
--num_of_classes 2 \
--classifier avgpooling_patch_reps \
--model_dir /work/HHRI-AI/YW/Yirong/CBraMod/models_weights/finetune-Stress_noleak_30chan_no400up_seed_siwen42-4review-default-avgpooling_reps 

python finetune_main.py \
--downstream_dataset CustomStress \
--datasets_dir /work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42 \
--num_of_classes 2 \
--classifier all_patch_reps_onelayer \
--model_dir /work/HHRI-AI/YW/Yirong/CBraMod/models_weights/finetune-Stress_noleak_30chan_no400up_seed_siwen42-4review-default-reps_onelayer 


python finetune_main.py \
--downstream_dataset CustomStress \
--datasets_dir /work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42 \
--num_of_classes 2 \
--classifier Labram_style_classifier \
--model_dir /work/HHRI-AI/YW/Yirong/CBraMod/models_weights/finetune-Stress_noleak_30chan_no400up_seed_siwen42-4review-Labram_style_classifier 


python finetune_main.py \
--downstream_dataset CustomStress \
--datasets_dir /work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42 \
--num_of_classes 2 \
--classifier Labram_style_classifier2 \
--model_dir /work/HHRI-AI/YW/Yirong/CBraMod/models_weights/finetune-Stress_noleak_30chan_no400up_seed_siwen42-4review-Labram_style_classifier2 
