"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

#from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import math
import re
import numpy as np
import os
import pickle
from collections import defaultdict
from downstream_dataset import KaggleERNLoader, CustomStressLoader, SEED7Loader, MotorLoader, SleepLoader
from downstream_dataset import _parse_subject_id
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


def _gather_stress_files_by_subject(root, expected_channels=None, valid_labels=None):
    """把 root 下 train/val/test 里所有 .pickle 汇总（忽略原始划分），过滤后按 subject 分组。
    返回 {subject_id: [abs_path, ...]}（每组已按 basename 排序）。

    这样 leave-one-subject-out 的划分完全由 subject 决定，与原始 train/val/test 无关。
    与 _gather_motor_files_by_subject 同构，只是 pickle 的键是 'X'/'y'（stress_preprocess.ipynb
    生成），不是 motor 用的 'signal'/'label'。
    """
    subdirs = [d for d in ["train", "val", "test"] if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        subdirs = ["."]

    seen_basenames = set()
    subj_to_paths = defaultdict(list)
    total = 0
    kept = 0
    skipped = 0
    label_skipped = 0
    for d in subdirs:
        folder = os.path.join(root, d)
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".pickle"):
                continue
            total += 1
            if f in seen_basenames:  # 去重（同名文件只保留一次）
                continue
            file_path = os.path.join(folder, f)
            try:
                sample = pickle.load(open(file_path, "rb"))
                if "X" not in sample or "y" not in sample:
                    skipped += 1
                    continue
                if expected_channels is not None and sample["X"].shape[0] != expected_channels:
                    skipped += 1
                    continue
                if valid_labels is not None:
                    try:
                        lbl_int = int(sample["y"])
                    except (TypeError, ValueError):
                        lbl_int = None
                    if lbl_int not in valid_labels:
                        label_skipped += 1
                        continue
            except Exception:
                skipped += 1
                continue
            subj = _parse_subject_id(f)
            if subj < 0:
                skipped += 1
                continue
            seen_basenames.add(f)
            subj_to_paths[subj].append(os.path.abspath(file_path))
            kept += 1

    for subj in subj_to_paths:
        subj_to_paths[subj] = sorted(subj_to_paths[subj], key=lambda p: os.path.basename(p))

    print(f"Stress LOSO - scanned {total} files across {subdirs}: kept {kept}, "
          f"skipped {skipped} (bad channel/missing), label_skipped {label_skipped}, "
          f"subjects found: {sorted(subj_to_paths.keys())}")
    return subj_to_paths


def prepare_STRESS_dataset_loso(root, fold, chan_size, is_instruct=False, eeg_max_len=-1, text_max_len=-1,
                                 n_folds=None, num_classes=2):
    """Leave-one-subject-out 划分（第 fold 折），与 prepare_motor_dataset_loso 同构。

    N = 被试数（stress 目前是 17 个，subject_edf_mapping.csv 里没有 Patient_ID=15）。
    test = subjects[fold]，val = subjects[(fold+1) % N]，train = 其余被试。
    只保留 label 在 [0, num_classes) 内的样本（Stress 为二分类：0=normal, 1=increase）。
    返回 (train_dataset, test_dataset, val_dataset, meta)。
    """
    valid_labels = set(range(num_classes))
    subj_to_paths = _gather_stress_files_by_subject(root, expected_channels=chan_size, valid_labels=valid_labels)
    subjects = sorted(subj_to_paths.keys())
    N = len(subjects)
    if N < 3:
        raise ValueError(f"LOSO 需要至少 3 个被试，实际只有 {N} 个: {subjects}")

    total_folds = N if n_folds is None else min(int(n_folds), N)
    if not (0 <= fold < total_folds):
        raise ValueError(f"fold={fold} 超出范围 [0, {total_folds})")

    test_subject = subjects[fold]
    val_subject = subjects[(fold + 1) % N]
    train_subjects = [s for s in subjects if s != test_subject and s != val_subject]

    train_files, val_files, test_files = [], [], []
    for s in train_subjects:
        train_files.extend(subj_to_paths[s])
    val_files.extend(subj_to_paths[val_subject])
    test_files.extend(subj_to_paths[test_subject])

    print(f"Stress LOSO fold {fold}/{total_folds} | N={N} | "
          f"test=S{test_subject} ({len(test_files)}), val=S{val_subject} ({len(val_files)}), "
          f"train={train_subjects} ({len(train_files)})")

    # root=None -> files 视为绝对路径（见 CustomStressLoader._resolve_path）
    train_dataset = CustomStressLoader(None, train_files, chan_size=chan_size, is_instruct=is_instruct,
                                        eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = CustomStressLoader(None, test_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True,
                                       eeg_max_len=eeg_max_len, text_max_len=text_max_len, return_sample_id=True)
    val_dataset = CustomStressLoader(None, val_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True,
                                      eeg_max_len=eeg_max_len, text_max_len=text_max_len, return_sample_id=True)

    meta = {
        'test_subject': int(test_subject),
        'val_subject': int(val_subject),
        'train_subjects': [int(s) for s in train_subjects],
        'n_folds': int(total_folds),
        'num_subjects': int(N),
        'subjects': [int(s) for s in subjects],
    }
    return train_dataset, test_dataset, val_dataset, meta


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
    print(f"Note: Neutral (label=2) samples will be filtered out, resulting in 6-class classification")

    # prepare training and test data loader
    train_dataset = SEED7Loader(os.path.join(root, "train"), train_files, chan_size=chan_size, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = SEED7Loader(os.path.join(root, "test"), test_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = SEED7Loader(os.path.join(root, "val"), val_files, chan_size=chan_size, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    
    print(f"SEED7 - After filtering neutral: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
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


def _gather_motor_files_by_subject(root, expected_channels=20, valid_labels=None):
    """把 root 下 train/val/test 里所有 .pickle 汇总（忽略原始划分），
    过滤到 20 通道，按 subject 分组。返回 {subject_id: [abs_path, ...]}（每组已按 basename 排序）。

    这样 leave-one-subject-out 的划分完全由 subject 决定，与原始 train/val/test 无关。

    valid_labels: 若给定（如 {0,1,2,3,4,5}），则只保留 label 在该集合内的样本。
      原始 pipeline 中 train 只喂 self.text[label]（label 必须是 0..num_classes-1），
      而 val/test 走 prompt 分支不查 label，因此原始 val/test 里可能混有超出类别范围的
      label（如 6）。LOSO 会把 val/test 的样本也放进 train，故必须在此统一过滤，
      否则训练时 self.text[label] 会 KeyError，评估时也会引入模型无法表示的类别。
    """
    subdirs = [d for d in ["train", "val", "test"] if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        # 兼容：root 本身直接放着 pickle
        subdirs = ["."]

    seen_basenames = set()
    subj_to_paths = defaultdict(list)
    total = 0
    kept = 0
    skipped = 0
    label_skipped = 0
    dropped_label_hist = defaultdict(int)
    for d in subdirs:
        folder = os.path.join(root, d)
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".pickle"):
                continue
            total += 1
            if f in seen_basenames:  # 去重（同名文件只保留一次）
                continue
            file_path = os.path.join(folder, f)
            try:
                sample = pickle.load(open(file_path, "rb"))
                if "signal" not in sample:
                    skipped += 1
                    continue
                if sample["signal"].shape[0] != expected_channels:
                    skipped += 1
                    continue
                # 标签范围过滤：超出类别范围的样本一律丢弃（train/val/test 一致）
                if valid_labels is not None:
                    lbl = sample.get("label", None)
                    try:
                        lbl_int = int(lbl)
                    except (TypeError, ValueError):
                        lbl_int = None
                    if lbl_int not in valid_labels:
                        label_skipped += 1
                        dropped_label_hist[lbl] += 1
                        continue
            except Exception as e:
                skipped += 1
                continue
            subj = _parse_subject_id(f)
            if subj < 0:
                skipped += 1
                if skipped <= 5:
                    print(f"  Skipping {f}: cannot parse subject id")
                continue
            seen_basenames.add(f)
            subj_to_paths[subj].append(os.path.abspath(file_path))
            kept += 1

    for subj in subj_to_paths:
        subj_to_paths[subj] = sorted(subj_to_paths[subj], key=lambda p: os.path.basename(p))

    print(f"Motor LOSO - scanned {total} files across {subdirs}: kept {kept}, "
          f"skipped {skipped} (bad channel/missing), label_skipped {label_skipped}, "
          f"subjects found: {sorted(subj_to_paths.keys())}")
    if valid_labels is not None and label_skipped > 0:
        print(f"Motor LOSO - dropped out-of-range labels (valid={sorted(valid_labels)}): "
              f"{dict(dropped_label_hist)}")
    return subj_to_paths


def prepare_motor_dataset_loso(root, fold, is_instruct=False, eeg_max_len=-1, text_max_len=-1,
                               n_folds=None, expected_channels=20, num_classes=6):
    """Leave-one-subject-out 划分（第 fold 折）。

    N = 被试数。test = subjects[fold]，val = subjects[(fold+1) % N]，train = 其余被试。
    只保留 label 在 [0, num_classes) 内的样本（Motor 为 6 类）。
    返回 (train_dataset, test_dataset, val_dataset, meta)。
    """
    valid_labels = set(range(num_classes))
    subj_to_paths = _gather_motor_files_by_subject(
        root, expected_channels=expected_channels, valid_labels=valid_labels)
    subjects = sorted(subj_to_paths.keys())
    N = len(subjects)
    if N < 3:
        raise ValueError(f"LOSO 需要至少 3 个被试，实际只有 {N} 个: {subjects}")

    total_folds = N if n_folds is None else min(int(n_folds), N)
    if not (0 <= fold < total_folds):
        raise ValueError(f"fold={fold} 超出范围 [0, {total_folds})")

    test_subject = subjects[fold]
    val_subject = subjects[(fold + 1) % N]
    train_subjects = [s for s in subjects if s != test_subject and s != val_subject]

    train_files, val_files, test_files = [], [], []
    for s in train_subjects:
        train_files.extend(subj_to_paths[s])
    val_files.extend(subj_to_paths[val_subject])
    test_files.extend(subj_to_paths[test_subject])

    print(f"Motor LOSO fold {fold}/{total_folds} | N={N} | "
          f"test=S{test_subject} ({len(test_files)}), val=S{val_subject} ({len(val_files)}), "
          f"train={train_subjects} ({len(train_files)})")

    # root=None -> files 视为绝对路径
    train_dataset = MotorLoader(None, train_files, is_instruct=is_instruct,
                                eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = MotorLoader(None, test_files, is_instruct=is_instruct, is_val=True,
                               eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = MotorLoader(None, val_files, is_instruct=is_instruct, is_val=True,
                              eeg_max_len=eeg_max_len, text_max_len=text_max_len)

    meta = {
        'test_subject': int(test_subject),
        'val_subject': int(val_subject),
        'train_subjects': [int(s) for s in train_subjects],
        'n_folds': int(total_folds),
        'num_subjects': int(N),
        'subjects': [int(s) for s in subjects],
    }
    return train_dataset, test_dataset, val_dataset, meta


def prepare_sleep_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    # Sleep data structure: sleep_data/train/, sleep_data/val/, sleep_data/test/
    # Each folder contains pickle files directly
    # Filter files to only include 6-channel data
    EXPECTED_CHANNELS = 6
    
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
    
    print(f"Sleep - Filtering files to only include {EXPECTED_CHANNELS}-channel data...")
    print(f"  Train: {len(all_train_files)} total files")
    train_files, train_skipped = filter_valid_files(train_folder, all_train_files)
    print(f"  Val: {len(all_val_files)} total files")
    val_files, val_skipped = filter_valid_files(val_folder, all_val_files)
    print(f"  Test: {len(all_test_files)} total files")
    test_files, test_skipped = filter_valid_files(test_folder, all_test_files)
    
    print(f"Sleep - After filtering: Train: {len(train_files)} (skipped {train_skipped}), Val: {len(val_files)} (skipped {val_skipped}), Test: {len(test_files)} (skipped {test_skipped})")

    # prepare training and test data loader
    train_dataset = SleepLoader(train_folder, train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = SleepLoader(test_folder, test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = SleepLoader(val_folder, val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
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