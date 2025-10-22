# Stress Dataset: BIOT, default, classifier, 16 chan 
# (data) root: "/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_16chan_no400up_swien42"
python run_binary_supervised.py \
--dataset CustomStress-16chan \
--n_classes 2 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--batch_size 512 \
--model BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# Stress Dataset: BIOT, Ada classifier, 30 chan 
# (data) root: "/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42"
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--batch_size 512 \
--model BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# Stress Dataset: STTransformer, 30 chan 
# (data) root: "/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42"
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--n_classes 2 \
--lr 5e-4 \
--in_channels 30 \
--sample_length 5 \
--batch_size 256 \
--model STTransformer 

# Stress Dataset: BIOT, Labram classifier, 16 chan 
# (data) root: "/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_16chan_no400up_swien42"
python run_binary_supervised.py \
--dataset CustomStress-16chan \
--n_classes 2 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--batch_size 512 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# Stress Dataset: BIOT, Labram classifier, 30 chan 
# (data) root: "/work/HHRI-AI/YW/Yirong/LaBramFinetune/augmented_data/Stress_noleak_30chan_no400up_seed_siwen42"
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--batch_size 512 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 
