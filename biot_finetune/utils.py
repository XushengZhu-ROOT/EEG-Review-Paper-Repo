import pickle
import torch
import numpy as np
import torch.nn.functional as F
import os
import re
from collections import defaultdict
from scipy.signal import resample
from scipy.signal import butter, iirnotch, filtfilt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


# ===== Stress: cross-model-stable sample_id (ported verbatim from
# labram_finetune/utils.py's compute_stress_sample_id / cbramod_finetune's
# custom_stress_dataset.py compute_stress_sample_id -- same regex/format so
# BIOT/ST produce identical sample_id strings to the other models for the
# same underlying chunk files). chunk_id looks like
# "Sub04_increase_edf27_chunk0012" (preprocessing/stress_preprocess.ipynb /
# stress_data/_run_*_preprocess.py's save_chunks()). =====
_STRESS_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(increase|normal)_edf(\d+)_chunk(\d+)$')
_STRESS_SUBJECT_RE = re.compile(r'(Sub\d+)_')


def compute_stress_sample_id(chunk_id):
    """'Sub04_increase_edf27_chunk0012' -> 'S04_edf27_chunk0012'."""
    m = _STRESS_SAMPLE_ID_RE.match(chunk_id)
    if not m:
        raise ValueError(f"Cannot parse chunk_id for sample_id: {chunk_id!r}")
    subject_num = int(m.group(1))
    edf_num = int(m.group(3))
    local_idx = int(m.group(4))
    return f"S{subject_num:02d}_edf{edf_num}_chunk{local_idx:04d}"


def extract_stress_subject_id(name):
    """'Sub04_increase_edf27_chunk0012.pickle' (or any path containing it) -> 'Sub04'."""
    m = _STRESS_SUBJECT_RE.search(os.path.basename(name))
    return m.group(1) if m else None


def list_stress_files_by_subject(root):
    """Scan root/{train,val,test}/*.pickle and group full paths by subject
    ('Sub04' -> [path, ...]), for building a subject-independent split."""
    subject_to_files = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".pickle"):
                continue
            sid = extract_stress_subject_id(fname)
            if sid is None:
                continue
            subject_to_files[sid].append(os.path.join(split_dir, fname))
    return subject_to_files


class StressLoader(torch.utils.data.Dataset):
    """Stress loader shared by both BIOT (200Hz, no filter -- preprocess_stress_biot)
    and STTransformer (250Hz, 4-40Hz band-pass -- preprocess_stress_sttransformer):
    both pipelines already resample to their own target rate, and both feed the
    model raw (channels, time) chunks (no reshape, unlike cbramod's own loader),
    so a single class parameterized by default_rate covers both -- same shape as
    TUABLoader but with subject/sample_id support for LOSO."""

    def __init__(self, root, files, sampling_rate=200, default_rate=200, return_sample_id=False):
        self.root = root
        self.files = files
        self.default_rate = default_rate
        self.sampling_rate = sampling_rate
        self.return_sample_id = return_sample_id

        if self.return_sample_id:
            self.sample_ids = [
                compute_stress_sample_id(os.path.splitext(os.path.basename(f))[0])
                for f in self.files
            ]
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError(
                    "Duplicate sample_id detected in StressLoader; check for "
                    "duplicate/conflicting chunk files."
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]

        if self.sampling_rate != self.default_rate:
            n_target = int(round(X.shape[-1] * self.sampling_rate / self.default_rate))
            X = resample(X, n_target, axis=-1)

        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(int(sample["y"]), dtype=torch.long)

        if self.return_sample_id:
            return X, Y, self.sample_ids[index]
        return X, Y


class KaggleERNLoader(torch.utils.data.Dataset):
    # [KaggleERN bestval] return_sample_id=False 时行为完全不变（现有训练/评估调用点
    # 不受影响）。True 时额外返回 sample_id（preprocess_KaggleERN_new.ipynb 存盘时
    # 写进 pickle 的 'epoch_id'，形如 "S02_Sess01_FB004"；找不到就退化成文件名本身），
    # 跟 finetune_evaluator.py/compute_metrics_from_npz.py 已经在用的 "^S(\d+)_" 被试号
    # 解析约定保持一致。
    def __init__(self, root, files, sampling_rate=200, return_sample_id=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.return_sample_id = return_sample_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["label"]
        X = torch.FloatTensor(X)
        if self.return_sample_id:
            sample_id = sample.get("epoch_id", os.path.splitext(os.path.basename(self.files[index]))[0])
            return X, Y, sample_id
        return X, Y

# ===== Motion: cross-model-stable sample_id (ported from
# cbramod/datasets/motortask_dataset.py's compute_sample_id, kept in sync
# so BIOT and CBraMod produce identical sample_id strings for the same
# underlying epoch file). epoch_id looks like "Sub04_Walkslow_epoch009".
_MOTION_SAMPLE_ID_TASK_ORDER = ['Walk', '8', 'Horizontal', 'Vertical', 'Pick', 'Stair']
_MOTION_SAMPLE_ID_SPEED_ORDER = ['slow', 'medium', 'fast']
_MOTION_SAMPLE_ID_TASK_OFFSET = 3000
_MOTION_SAMPLE_ID_SPEED_OFFSET = 1000
_MOTION_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(.+?)_epoch(\d+)$')
_MOTION_SUBJECT_RE = re.compile(r'(Sub\d+)_')


def _parse_motion_task_token(task_token):
    for speed in _MOTION_SAMPLE_ID_SPEED_ORDER:
        if task_token.endswith(speed):
            return task_token[: -len(speed)], speed
    raise ValueError(f"Cannot parse speed suffix (slow/medium/fast) from task token: {task_token!r}")


def compute_motion_sample_id(epoch_id):
    """Deterministic sample_id "S{subject:02d}_ep{index:05d}" from an
    epoch_id like 'Sub04_Walkslow_epoch009'. Pure function of the string
    itself, independent of shuffle/batch_size/num_workers, and identical
    across models trained on the same underlying epoch files."""
    m = _MOTION_SAMPLE_ID_RE.match(epoch_id)
    if not m:
        raise ValueError(f"Cannot parse epoch_id for sample_id: {epoch_id!r}")
    subject_num = int(m.group(1))
    task_token = m.group(2)
    local_idx = int(m.group(3))
    base_task, speed = _parse_motion_task_token(task_token)
    if base_task not in _MOTION_SAMPLE_ID_TASK_ORDER:
        raise ValueError(
            f"Unknown base task {base_task!r} parsed from epoch_id {epoch_id!r}; "
            f"expected one of {_MOTION_SAMPLE_ID_TASK_ORDER}"
        )
    task_idx = _MOTION_SAMPLE_ID_TASK_ORDER.index(base_task)
    speed_idx = _MOTION_SAMPLE_ID_SPEED_ORDER.index(speed)
    global_index = task_idx * _MOTION_SAMPLE_ID_TASK_OFFSET + speed_idx * _MOTION_SAMPLE_ID_SPEED_OFFSET + local_idx
    if global_index > 99999:
        raise ValueError(f"sample_id index overflow (>99999) for epoch_id {epoch_id!r}: {global_index}")
    return f"S{subject_num:02d}_ep{global_index:05d}"


def extract_motion_subject_id(name):
    """'Sub04_8fast_epoch001.pickle' (or any path containing it) -> 'Sub04'."""
    m = _MOTION_SUBJECT_RE.search(os.path.basename(name))
    return m.group(1) if m else None


def list_motion_files_by_subject(root):
    """Scan root/{train,val,test}/*.pickle and group full paths by subject
    ('Sub04' -> [path, ...]), for building a subject-independent split."""
    subject_to_files = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".pickle"):
                continue
            sid = extract_motion_subject_id(fname)
            if sid is None:
                continue
            subject_to_files[sid].append(os.path.join(split_dir, fname))
    return subject_to_files


class MotionLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, in_channels=16, return_sample_id=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.in_channels = in_channels  # <<< 必須加上
        self.return_sample_id = return_sample_id

        # 20 → 16 channel mapping
        self.MOTION_16CH = [1,2,4,6,7,17,14,12,15,16,0,3,13,18,5,8]

        # sample_id 在构造时一次性算好，跟 self.files 一一对应；
        # 与 shuffle / batch_size / num_workers 无关，解析失败直接报错。
        self.sample_ids = [
            compute_motion_sample_id(os.path.splitext(os.path.basename(f))[0])
            for f in self.files
        ]
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(
                "Duplicate sample_id detected in MotionLoader; check for "
                "duplicate/conflicting epoch files."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        for _ in range(len(self.files)):
            sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
            X = sample["signal"]

            # --- 20ch → 16ch ---
            X = X[self.MOTION_16CH, :]

            # --- 跳過不是16ch的資料（理論上不會發生） ---
            if X.shape[0] != self.in_channels:
                index = (index + 1) % len(self.files)
                continue

            # --- resample (if needed) ---
            if self.sampling_rate != self.default_rate:
                X = resample(X, 10 * self.sampling_rate, axis=-1)

            # --- 95% quantile normalization ---
            X = X / (
                np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )

            # --- convert dtype ---
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(sample["label"], dtype=torch.long)

            if self.return_sample_id:
                return X, Y, self.sample_ids[index]
            return X, Y

        raise RuntimeError(
            f"No valid sample with in_channels={self.in_channels} found."
        )


class MotionSTLoader(torch.utils.data.Dataset):
    """Motion loader for the STTransformer path (Motiondata_ST, native 20ch
    @ 250Hz). Unlike MotionLoader, which hard-selects 16 of 20 channels to
    match BIOT's 16-channel pretrained checkpoint, STTransformer trains from
    scratch with no channel-count constraint -- so this keeps all native
    channels, mirroring how Sleep/SEED already feed STTransformer their full
    native channel count. Kept as a separate class so MotionLoader (BIOT)
    stays untouched. Same sample_id scheme as MotionLoader so BIOT and ST
    produce identical sample_id sets for the same underlying epoch files."""

    def __init__(self, root, files, sampling_rate=250, in_channels=20, return_sample_id=False):
        self.root = root
        self.files = files
        self.default_rate = 250
        self.sampling_rate = sampling_rate
        self.in_channels = in_channels
        self.return_sample_id = return_sample_id

        self.sample_ids = [
            compute_motion_sample_id(os.path.splitext(os.path.basename(f))[0])
            for f in self.files
        ]
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(
                "Duplicate sample_id detected in MotionSTLoader; check for "
                "duplicate/conflicting epoch files."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]

        if X.shape[0] != self.in_channels:
            raise ValueError(
                f"MotionSTLoader: expected {self.in_channels} channels, got "
                f"{X.shape[0]} in {self.files[index]!r}"
            )

        if self.sampling_rate != self.default_rate:
            n_target = int(round(X.shape[-1] * self.sampling_rate / self.default_rate))
            X = resample(X, n_target, axis=-1)

        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(sample["label"], dtype=torch.long)

        if self.return_sample_id:
            return X, Y, self.sample_ids[index]
        return X, Y


class SEEDLoader(torch.utils.data.Dataset):
    """
    SEED数据集加载器
    数据格式: pickle文件，包含 'signal' (62, time_points), 'label' (int 0-6), 'epoch_id' (str)
    
    根据process_seed.ipynb的预处理流程：
    - BIOT: 数据已resample到200Hz，4秒数据=800个时间点，label已经是0-6
    - STTransformer: 数据已resample到250Hz，4秒数据=1000个时间点，label已经是0-6
    - 数据已经过预处理，不需要重采样和label转换
    - 只需要95%分位数归一化（BIOT预处理中注释掉了，但通常训练时需要）
    
    标签映射（移除neutral，从7分类变为6分类）：
    - 原始: happy=0, sad=1, neutral=2, disgust=3, fear=4, surprise=5, anger=6
    - 新的: happy=0, sad=1, disgust=2, fear=3, surprise=4, anger=5 (跳过neutral=2)
    """
    def __init__(self, root, files, sampling_rate=200, return_epoch_id=False):
        self.root = root
        self.files = files
        self.sampling_rate = sampling_rate  # 用于验证，数据已经是200Hz或250Hz
        self.return_epoch_id = return_epoch_id  # 是否返回epoch_id（用于评估）
        
        # 标签重新映射：跳过neutral (2)，将其他标签映射到0-5
        # 0->0, 1->1, 2->跳过, 3->2, 4->3, 5->4, 6->5
        self.label_map = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 处理文件路径：files可能包含子目录路径
        file_path = self.files[index]
        if os.path.isabs(file_path) or (self.root and file_path.startswith(self.root)):
            full_path = file_path
        else:
            full_path = os.path.join(self.root, file_path) if self.root else file_path
        
        sample = pickle.load(open(full_path, "rb"))
        X = sample["signal"]  # shape: (62, time_points) - 已经是200Hz或250Hz处理后的数据
        
        # 数据已经预处理过，不需要重采样
        # 只需要95%分位数归一化（BIOT预处理中注释掉了，但训练时通常需要）
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        
        # 转换数据类型
        X = torch.FloatTensor(X)
        
        # 重新映射标签：跳过neutral (2)，将其他标签映射到0-5
        original_label = int(sample["label"])
        if original_label == 2:  # neutral，不应该出现（已在prepare_SEED_dataloader中过滤）
            raise ValueError(f"Unexpected neutral label (2) found in file: {full_path}")
        Y = self.label_map[original_label]
        Y = torch.tensor(Y, dtype=torch.long)
        
        # 如果需要返回epoch_id（用于评估），则返回三元组
        if self.return_epoch_id:
            epoch_id = sample.get("epoch_id", "")
            return X, Y, epoch_id
        
        return X, Y

class SleepLoader(torch.utils.data.Dataset):
    """
    Sleep数据集加载器
    数据格式: pickle文件，包含 'signal' (6, 7500), 'label' (int 0-4), 'epoch_id' (str)
    
    数据特性：
    - 6个通道：['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
    - 5个类别：0, 1, 2, 3, 4
    - 30秒数据，采样率250Hz，所以是7500个时间点
    - 数据已经预处理过，只需要95%分位数归一化
    """
    def __init__(self, root, files, sampling_rate=250, return_epoch_id=False):
        self.root = root
        self.files = files
        self.sampling_rate = sampling_rate  # 用于验证，数据已经是250Hz
        self.return_epoch_id = return_epoch_id  # 是否返回epoch_id（用于评估）

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 处理文件路径：files可能包含子目录路径
        file_path = self.files[index]
        if os.path.isabs(file_path) or (self.root and file_path.startswith(self.root)):
            full_path = file_path
        else:
            full_path = os.path.join(self.root, file_path) if self.root else file_path
        
        sample = pickle.load(open(full_path, "rb"))
        X = sample["signal"]  # shape: (6, 7500) - 已经是250Hz处理后的数据
        
        # 数据已经预处理过，不需要重采样
        # 只需要95%分位数归一化
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        
        # 转换数据类型
        X = torch.FloatTensor(X)
        
        # 标签已经是0-4，直接使用
        Y = int(sample["label"])
        Y = torch.tensor(Y, dtype=torch.long)
        
        # 如果需要返回epoch_id（用于评估），则返回三元组
        if self.return_epoch_id:
            epoch_id = sample.get("epoch_id", "")
            return X, Y, epoch_id
        
        return X, Y

class BiotSleepLoader(torch.utils.data.Dataset):
    """
    BIOT Sleep数据集加载器（用于200Hz采样率）
    数据格式: pickle文件，包含 'signal' (6, 6000), 'label' (int 0-4), 'epoch_id' (str)
    
    数据特性：
    - 6个通道：['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
    - 5个类别：0, 1, 2, 3, 4
    - 30秒数据，采样率200Hz，所以是6000个时间点
    - 数据已经预处理过，只需要95%分位数归一化
    """
    def __init__(self, root, files, sampling_rate=200, return_epoch_id=False):
        self.root = root
        self.files = files
        self.sampling_rate = sampling_rate  # 用于验证，数据已经是200Hz
        self.return_epoch_id = return_epoch_id  # 是否返回epoch_id（用于评估）

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 处理文件路径：files可能包含子目录路径
        file_path = self.files[index]
        if os.path.isabs(file_path) or (self.root and file_path.startswith(self.root)):
            full_path = file_path
        else:
            full_path = os.path.join(self.root, file_path) if self.root else file_path
        
        sample = pickle.load(open(full_path, "rb"))
        X = sample["signal"]  # shape: (6, 6000) - 已经是200Hz处理后的数据
        
        # 数据已经预处理过，不需要重采样
        # 只需要95%分位数归一化
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        
        # 转换数据类型
        X = torch.FloatTensor(X)
        
        # 标签已经是0-4，直接使用
        Y = int(sample["label"])
        Y = torch.tensor(Y, dtype=torch.long)
        
        # 如果需要返回epoch_id（用于评估），则返回三元组
        if self.return_epoch_id:
            epoch_id = sample.get("epoch_id", "")
            return X, Y, epoch_id
        
        return X, Y
        
class CHBMITLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # 2560 -> 2000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class PTBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=500):
        self.root = root
        self.files = files
        self.default_rate = 500
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self.freq * 5, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y


class HARLoader(torch.utils.data.Dataset):
    def __init__(self, dir, list_IDs, sampling_rate=50):
        self.list_IDs = list_IDs
        self.dir = dir
        self.label_map = ["1", "2", "3", "4", "5", "6"]
        self.default_rate = 50
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, "rb"))
        X, y = sample["X"], self.label_map.index(sample["y"])
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(2.56 * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(
                np.abs(X), q=0.95, interpolation="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        return torch.FloatTensor(X), y


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, root_prest, root_shhs):

        # prest dataset
        self.root_prest = root_prest
        exception_files = ["319431_data.npy"]
        self.prest_list = list(
            filter(
                lambda x: ("data" in x) and (x not in exception_files),
                os.listdir(self.root_prest),
            )
        )

        PREST_LENGTH = 2000
        WINDOW_SIZE = 200

        print("(prest) unlabeled data size:", len(self.prest_list) * 16)
        self.prest_idx_all = np.arange(PREST_LENGTH // WINDOW_SIZE)
        self.prest_mask_idx_N = PREST_LENGTH // WINDOW_SIZE // 3

        SHHS_LENGTH = 6000
        # shhs dataset
        self.root_shhs = root_shhs
        self.shhs_list = os.listdir(self.root_shhs)
        print("(shhs) unlabeled data size:", len(self.shhs_list))
        self.shhs_idx_all = np.arange(SHHS_LENGTH // WINDOW_SIZE)
        self.shhs_mask_idx_N = SHHS_LENGTH // WINDOW_SIZE // 5

    def __len__(self):
        return len(self.prest_list) + len(self.shhs_list)

    def prest_load(self, index):
        sample_path = self.prest_list[index]
        # (16, 16, 2000), 10s
        samples = np.load(os.path.join(self.root_prest, sample_path)).astype("float32")

        # find all zeros or all 500 signals and then remove them
        samples_max = np.max(samples, axis=(1, 2))
        samples_min = np.min(samples, axis=(1, 2))
        valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        samples = samples[valid]

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0

    def shhs_load(self, index):
        sample_path = self.shhs_list[index]
        # (2, 3750) sampled at 125
        sample = pickle.load(open(os.path.join(self.root_shhs, sample_path), "rb"))
        # (2, 6000) resample to 200
        samples = resample(sample, 6000, axis=-1)

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # generate samples and targets and mask_indices
        samples = torch.FloatTensor(samples)

        return samples, 1

    def __getitem__(self, index):
        if index < len(self.prest_list):
            return self.prest_load(index)
        else:
            index = index - len(self.prest_list)
            return self.shhs_load(index)


def collate_fn_unsupervised_pretrain(batch):
    prest_samples, shhs_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            prest_samples.append(sample)
        else:
            shhs_samples.append(sample)

    shhs_samples = torch.stack(shhs_samples, 0)
    if len(prest_samples) > 0:
        prest_samples = torch.cat(prest_samples, 0)
        return prest_samples, shhs_samples
    return 0, shhs_samples


def collate_fn_seed_with_epoch_id(batch):
    """
    自定义collate函数，用于处理SEED数据集返回(X, Y, epoch_id)的情况
    """
    # batch是list of tuples: [(X1, Y1, epoch_id1), (X2, Y2, epoch_id2), ...]
    X_list, Y_list, epoch_id_list = zip(*batch)
    
    # 堆叠X和Y
    X_batch = torch.stack(X_list, dim=0)
    Y_batch = torch.stack(Y_list, dim=0)
    
    # epoch_id保持为list
    epoch_id_batch = list(epoch_id_list)
    
    return X_batch, Y_batch, epoch_id_batch

def collate_fn_sleep_with_epoch_id(batch):
    """
    自定义collate函数，用于处理Sleep数据集返回(X, Y, epoch_id)的情况
    """
    # batch是list of tuples: [(X1, Y1, epoch_id1), (X2, Y2, epoch_id2), ...]
    X_list, Y_list, epoch_id_list = zip(*batch)
    
    # 堆叠X和Y
    X_batch = torch.stack(X_list, dim=0)
    Y_batch = torch.stack(Y_list, dim=0)
    
    # epoch_id保持为list
    epoch_id_batch = list(epoch_id_list)
    
    return X_batch, Y_batch, epoch_id_batch

def collate_fn_biot_sleep_with_epoch_id(batch):
    """
    自定义collate函数，用于处理BiotSleep数据集返回(X, Y, epoch_id)的情况
    """
    # batch是list of tuples: [(X1, Y1, epoch_id1), (X2, Y2, epoch_id2), ...]
    X_list, Y_list, epoch_id_list = zip(*batch)
    
    # 堆叠X和Y
    X_batch = torch.stack(X_list, dim=0)
    Y_batch = torch.stack(Y_list, dim=0)
    
    # epoch_id保持为list
    epoch_id_batch = list(epoch_id_list)
    
    return X_batch, Y_batch, epoch_id_batch

def collate_fn_motion_with_sample_id(batch):
    """
    自定义collate函数，用于处理MotionLoader(return_sample_id=True)返回(X, Y, sample_id)的情况
    """
    X_list, Y_list, sample_id_list = zip(*batch)
    X_batch = torch.stack(X_list, dim=0)
    Y_batch = torch.stack(Y_list, dim=0)
    sample_id_batch = list(sample_id_list)
    return X_batch, Y_batch, sample_id_batch

def collate_fn_stress_with_sample_id(batch):
    """
    自定义collate函数，用于处理StressLoader(return_sample_id=True)返回(X, Y, sample_id)的情况
    """
    X_list, Y_list, sample_id_list = zip(*batch)
    X_batch = torch.stack(X_list, dim=0)
    Y_batch = torch.stack(Y_list, dim=0)
    sample_id_batch = list(sample_id_list)
    return X_batch, Y_batch, sample_id_batch


def collate_fn_kaggleern_with_sample_id(batch):
    """
    [KaggleERN bestval] 自定义collate函数，用于处理KaggleERNLoader(return_sample_id=True)
    返回(X, Y, sample_id)的情况。Y 在 KaggleERNLoader 里是原始 label（不是 tensor），
    跟 collate_fn_stress_with_sample_id 不同，这里用 torch.tensor 而不是 torch.stack。
    """
    X_list, Y_list, sample_id_list = zip(*batch)
    X_batch = torch.stack(X_list, dim=0)
    Y_batch = torch.tensor(Y_list, dtype=torch.float32)
    sample_id_batch = list(sample_id_list)
    return X_batch, Y_batch, sample_id_batch

class EEGSupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, tuev_data, chb_mit_data, iiic_data, tuab_data):
        # for TUEV
        tuev_root, tuev_files = tuev_data
        self.tuev_root = tuev_root
        self.tuev_files = tuev_files
        self.tuev_size = len(self.tuev_files)

        # for CHB-MIT
        chb_mit_root, chb_mit_files = chb_mit_data
        self.chb_mit_root = chb_mit_root
        self.chb_mit_files = chb_mit_files
        self.chb_mit_size = len(self.chb_mit_files)

        # for IIIC seizure
        iiic_x, iiic_y = iiic_data
        self.iiic_x = iiic_x
        self.iiic_y = iiic_y
        self.iiic_size = len(self.iiic_x)

        # for TUAB
        tuab_root, tuab_files = tuab_data
        self.tuab_root = tuab_root
        self.tuab_files = tuab_files
        self.tuab_size = len(self.tuab_files)

    def __len__(self):
        return self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size

    def tuev_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuev_root, self.tuev_files[index]), "rb")
        )
        X = sample["signal"]
        # 256 * 5 -> 1000
        X = resample(X, 1000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y, 0

    def chb_mit_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.chb_mit_root, self.chb_mit_files[index]), "rb")
        )
        X = sample["X"]
        # 2560 -> 2000
        X = resample(X, 2000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 1

    def iiic_load(self, index):
        data = self.iiic_x[index]
        samples = torch.FloatTensor(data)
        samples = samples / (
            torch.quantile(torch.abs(samples), q=0.95, dim=-1, keepdim=True) + 1e-8
        )
        y = np.argmax(self.iiic_y[index])
        return samples, y, 2

    def tuab_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuab_root, self.tuab_files[index]), "rb")
        )
        X = sample["X"]
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 3

    def __getitem__(self, index):
        if index < self.tuev_size:
            return self.tuev_load(index)
        elif index < self.tuev_size + self.chb_mit_size:
            index = index - self.tuev_size
            return self.chb_mit_load(index)
        elif index < self.tuev_size + self.chb_mit_size + self.iiic_size:
            index = index - self.tuev_size - self.chb_mit_size
            return self.iiic_load(index)
        elif (
            index < self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size
        ):
            index = index - self.tuev_size - self.chb_mit_size - self.iiic_size
            return self.tuab_load(index)
        else:
            raise ValueError("index out of range")


def collate_fn_supervised_pretrain(batch):
    tuev_samples, tuev_labels = [], []
    iiic_samples, iiic_labels = [], []
    chb_mit_samples, chb_mit_labels = [], []
    tuab_samples, tuab_labels = [], []

    for sample, labels, idx in batch:
        if idx == 0:
            tuev_samples.append(sample)
            tuev_labels.append(labels)
        elif idx == 1:
            iiic_samples.append(sample)
            iiic_labels.append(labels)
        elif idx == 2:
            chb_mit_samples.append(sample)
            chb_mit_labels.append(labels)
        elif idx == 3:
            tuab_samples.append(sample)
            tuab_labels.append(labels)
        else:
            raise ValueError("idx out of range")

    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
        tuev_labels = torch.LongTensor(tuev_labels)
    if len(iiic_samples) > 0:
        iiic_samples = torch.stack(iiic_samples)
        iiic_labels = torch.LongTensor(iiic_labels)
    if len(chb_mit_samples) > 0:
        chb_mit_samples = torch.stack(chb_mit_samples)
        chb_mit_labels = torch.LongTensor(chb_mit_labels)
    if len(tuab_samples) > 0:
        tuab_samples = torch.stack(tuab_samples)
        tuab_labels = torch.LongTensor(tuab_labels)

    return (
        (tuev_samples, tuev_labels),
        (iiic_samples, iiic_labels),
        (chb_mit_samples, chb_mit_labels),
        (tuab_samples, tuab_labels),
    )


# define focal loss on binary classification
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


# define binary cross entropy loss
def BCE(y_hat, y, pos_weight=None):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss_per_sample = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    
    if pos_weight is not None:
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, device=y_hat.device, dtype=y_hat.dtype)
        else:
            pos_weight = pos_weight.to(y_hat.device, dtype=y_hat.dtype)

        # - 當 y == 1.0 (正樣本) 時, 權重為 pos_weight
        # - 當 y == 0.0 (負樣本) 時, 權重為 1.0
        w = torch.where(y == 1.0, pos_weight, 1.0)
        
        loss_per_sample = w * loss_per_sample
    return loss_per_sample.mean()
