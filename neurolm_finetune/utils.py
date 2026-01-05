"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

#from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import math
import numpy as np
import os
import pickle
from downstream_dataset import TUABLoader, TUEVLoader, TUSLLoader, HMCLoader, WorkloadLoader, KaggleERNLoader, CustomStressLoader, SEED7Loader, MotorLoader
from metrics import binary_metrics_fn, multiclass_metrics_fn


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def prepare_TUEV_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUSL_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUSLLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUSLLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUSLLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_HMC_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = HMCLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = HMCLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = HMCLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_Workload_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = WorkloadLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = WorkloadLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = WorkloadLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_KaggleERN_dataset(root, chan_size, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = KaggleERNLoader(os.path.join(root, "train"), train_files, chan_size=chan_size, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = KaggleERNLoader(os.path.join(root, "test"), test_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = KaggleERNLoader(os.path.join(root, "val"), val_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_STRESS_dataset(root, chan_size, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = CustomStressLoader(os.path.join(root, "train"), train_files, chan_size=chan_size, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = CustomStressLoader(os.path.join(root, "test"), test_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = CustomStressLoader(os.path.join(root, "val"), val_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_SEED7_dataset(root, chan_size, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    # SEED7 data structure: seed_data/train/subject_X/, seed_data/val/subject_X/, seed_data/test/subject_X/
    # Each subject folder contains pickle files
    train_subjects = os.listdir(os.path.join(root, "train"))
    val_subjects = os.listdir(os.path.join(root, "val"))
    test_subjects = os.listdir(os.path.join(root, "test"))
    
    train_files = []
    for subject in train_subjects:
        subject_path = os.path.join(root, "train", subject)
        if os.path.isdir(subject_path):
            files = [os.path.join(subject, f) for f in os.listdir(subject_path) if f.endswith('.pickle')]
            train_files.extend(files)
    
    val_files = []
    for subject in val_subjects:
        subject_path = os.path.join(root, "val", subject)
        if os.path.isdir(subject_path):
            files = [os.path.join(subject, f) for f in os.listdir(subject_path) if f.endswith('.pickle')]
            val_files.extend(files)
    
    test_files = []
    for subject in test_subjects:
        subject_path = os.path.join(root, "test", subject)
        if os.path.isdir(subject_path):
            files = [os.path.join(subject, f) for f in os.listdir(subject_path) if f.endswith('.pickle')]
            test_files.extend(files)

    print(f"SEED7 - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # prepare training and test data loader
    train_dataset = SEED7Loader(os.path.join(root, "train"), train_files, chan_size=chan_size, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = SEED7Loader(os.path.join(root, "test"), test_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = SEED7Loader(os.path.join(root, "val"), val_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    return train_dataset, test_dataset, val_dataset


def prepare_motor_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    # Motor data structure: motor_data_20channels/train/, motor_data_20channels/val/, motor_data_20channels/test/
    # Each folder contains pickle files directly
    # Filter files to only include 20-channel data
    EXPECTED_CHANNELS = 20
    
    def filter_valid_files(folder_path, file_list):
        """Filter files to only include those with exactly EXPECTED_CHANNELS channels"""
        valid_files = []
        skipped_count = 0
        for f in file_list:
            file_path = os.path.join(folder_path, f)
            try:
                sample = pickle.load(open(file_path, "rb"))
                if "signal" in sample:
                    signal = sample["signal"]
                    if signal.shape[0] == EXPECTED_CHANNELS:
                        valid_files.append(f)
                    else:
                        skipped_count += 1
                        if skipped_count <= 5:  # Only print first 5 skipped files
                            print(f"  Skipping {f}: expected {EXPECTED_CHANNELS} channels, got {signal.shape[0]}")
                else:
                    skipped_count += 1
                    if skipped_count <= 5:
                        print(f"  Skipping {f}: missing 'signal' key")
            except Exception as e:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"  Skipping {f}: error loading file - {str(e)}")
        if skipped_count > 5:
            print(f"  ... and {skipped_count - 5} more files skipped")
        return valid_files, skipped_count
    
    train_folder = os.path.join(root, "train")
    val_folder = os.path.join(root, "val")
    test_folder = os.path.join(root, "test")
    
    all_train_files = [f for f in os.listdir(train_folder) if f.endswith('.pickle')]
    all_val_files = [f for f in os.listdir(val_folder) if f.endswith('.pickle')]
    all_test_files = [f for f in os.listdir(test_folder) if f.endswith('.pickle')]
    
    print(f"Motor - Filtering files to only include {EXPECTED_CHANNELS}-channel data...")
    print(f"  Train: {len(all_train_files)} total files")
    train_files, train_skipped = filter_valid_files(train_folder, all_train_files)
    print(f"  Val: {len(all_val_files)} total files")
    val_files, val_skipped = filter_valid_files(val_folder, all_val_files)
    print(f"  Test: {len(all_test_files)} total files")
    test_files, test_skipped = filter_valid_files(test_folder, all_test_files)
    
    print(f"Motor - After filtering: Train: {len(train_files)} (skipped {train_skipped}), Val: {len(val_files)} (skipped {val_skipped}), Test: {len(test_files)} (skipped {test_skipped})")

    # prepare training and test data loader
    train_dataset = MotorLoader(train_folder, train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = MotorLoader(test_folder, test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = MotorLoader(val_folder, val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    return train_dataset, test_dataset, val_dataset


def get_metrics(output, target, metrics, is_binary):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results
