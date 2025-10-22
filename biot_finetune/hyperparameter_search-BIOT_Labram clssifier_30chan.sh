
# --- Labram style calssifier: grid search -----
# lr: {1e−3, 5e−4, 1e−4, 5e−5}
# weight_decay: {1e−5, 1e−4, 1e−3, 5e−2}
# batch size: {256, 512, 1024}

# 5e−4, 1e−5, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.0005 \
--weight_decay 0.00001 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 5e−4, 1e−4, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.0005 \
--weight_decay 0.0001 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 


# 5e−4, 1e−3, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.0005 \
--weight_decay 0.001 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 5e−4, 5e−2, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.0005 \
--weight_decay 0.00002 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# ---

# 1e−3, 1e−4, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.0001 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 


# 1e-3, 1e−3, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.001 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 1e-3, 5e−2, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.00002 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# ---

# 1e−4, 1e−5, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.0001 \
--weight_decay 0.00001 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 1e−4, 1e−4, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.0001 \
--weight_decay 0.0001 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 


# 1e−4, 1e−3, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.0001 \
--weight_decay 0.001 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 1e−4, 5e−2, 512
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.0001 \
--weight_decay 0.00002 \
--batch_size 512 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# ---

# ---

# 1e−3, 1e−5, 1024
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.00001 \
--batch_size 1024 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 1e−3, 1e−4, 1024
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.0001 \
--batch_size 1024 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 


# 1e-3, 1e−3, 1024
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.001 \
--batch_size 1024 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 1e-3, 5e−2, 1024
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.00002 \
--batch_size 1024 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# --

# 1e−3, 1e−5, 256
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.00001 \
--batch_size 256 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 1e−3, 1e−4, 256
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.0001 \
--batch_size 256 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 


# 1e-3, 1e−3, 256
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.001 \
--batch_size 256 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 

# 1e-3, 5e−2, 256
python run_binary_supervised.py \
--dataset CustomStress-30chan \
--lr 0.001 \
--weight_decay 0.00002 \
--batch_size 256 \
--n_classes 2 \
--dataset_channels 30 \
--in_channels 16 \
--sampling_rate 200 \
--token_size 200 \
--hop_length 100 \
--sample_length 5 \
--model LabramClassifier-BIOT \
--pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt 
