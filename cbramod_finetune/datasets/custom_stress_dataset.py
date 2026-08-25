import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import re
import random
import lmdb
import pickle
from scipy import signal
from collections import defaultdict


def extract_subject_id_from_name(name):
    """从文件名中提取受试者 ID，例如 Sub04_increase_edf27_chunk0012 -> Sub04"""
    basename = os.path.basename(name)
    match = re.match(r'(Sub\d+)_', basename)
    if match:
        return match.group(1)
    return None


# chunk_id 形如 'Sub04_increase_edf27_chunk0012'（preprocessing/stress_preprocess.ipynb 生成）
_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(increase|normal)_edf(\d+)_chunk(\d+)$')


def compute_stress_sample_id(chunk_id):
    """
    由 chunk_id 确定性地生成 sample_id：S{subject:02d}_edf{edf_num}_chunk{local_idx:04d}。
    纯函数，只依赖文件名本身，与 shuffle / batch_size / num_workers 无关。
    """
    m = _SAMPLE_ID_RE.match(chunk_id)
    if not m:
        raise ValueError(f"Cannot parse chunk_id for sample_id: {chunk_id!r}")
    subject_num = int(m.group(1))
    edf_num = int(m.group(3))
    local_idx = int(m.group(4))
    return f"S{subject_num:02d}_edf{edf_num}_chunk{local_idx:04d}"


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            channel_size=30,
            window_size=5,
            # 若不为 None，忽略 data_dir/mode 目录扫描，直接使用该文件列表（用于受试者独立划分）。
            # 旧的 random-epoch 路径不传，行为与修改前完全一致。
            file_list=None,
            # 是否额外返回 (epoch_id, sample_id)。默认 False，保持旧调用点字节不变；
            # 旧版本 stress 数据（chunk_{i}.pickle 命名，无 subject id）不支持 True。
            return_sample_id=False,
    ):
        super(CustomDataset, self).__init__()
        self.channel_size = channel_size
        self.window_size = window_size
        self.return_sample_id = return_sample_id

        if file_list is not None:
            self.files = list(file_list)
        else:
            self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]

        if self.return_sample_id:
            self.sample_ids = [
                compute_stress_sample_id(os.path.splitext(os.path.basename(f))[0]) for f in self.files
            ]
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError("Duplicate sample_id detected; check for duplicate/conflicting chunk files.")

    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['X']
        # print("Shape of data[X]: ", data.shape)
        label = data_dict['y']
        # data = signal.resample(data, 2000, axis=-1)
        data = data.reshape(self.channel_size, self.window_size, 200) # for 30 channels
        if self.return_sample_id:
            epoch_id = os.path.splitext(os.path.basename(file))[0]
            sample_id = self.sample_ids[idx]
            return data/100, label, epoch_id, sample_id
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        if self.return_sample_id:
            epoch_ids = [x[2] for x in batch]
            sample_ids = [x[3] for x in batch]
            return to_tensor(x_data), to_tensor(y_label), epoch_ids, sample_ids
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.channel_size = params.channel_size
        self.window_size = params.window_size

    def get_data_loader(self):
        """
        根据 params.split_mode 选择划分策略：
          - random_epoch（默认/旧方法）：直接读已有的 train/val/test 子目录
          - subject_independent（LOSO 新增）：按受试者严格划分，train/val/test 互不重叠
        """
        split_mode = getattr(self.params, 'split_mode', 'random_epoch')
        if split_mode == 'subject_independent':
            return self._get_data_loader_subject_independent()
        return self._get_data_loader_random_epoch()

    def _get_data_loader_random_epoch(self):
        """[旧方法] 使用预先按 epoch 随机划分好的 train/val/test 目录。"""
        train_set = CustomDataset(self.datasets_dir, mode='train', channel_size=self.channel_size, window_size=self.window_size)
        val_set = CustomDataset(self.datasets_dir, mode='val', channel_size=self.channel_size, window_size=self.window_size)
        test_set = CustomDataset(self.datasets_dir, mode='test', channel_size=self.channel_size, window_size=self.window_size)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        return self._build_loaders(train_set, val_set, test_set)

    def _collect_all_pickle_files(self):
        """从 train/val/test 三个子目录收集全部 pickle。每个受试者都可能出现在三个目录中；
        受试者独立评估时需要先合并再按 Subject ID 重新划分。"""
        all_files = []
        for mode in ['train', 'val', 'test']:
            mode_dir = os.path.join(self.datasets_dir, mode)
            if not os.path.isdir(mode_dir):
                continue
            for fname in os.listdir(mode_dir):
                if fname.endswith('.pickle'):
                    all_files.append(os.path.join(mode_dir, fname))
        return all_files

    def _group_files_by_subject(self, all_files):
        """按受试者 ID 分组文件路径。"""
        subject_to_files = defaultdict(list)
        skipped = 0
        for fpath in all_files:
            sid = extract_subject_id_from_name(fpath)
            if sid is None:
                skipped += 1
                continue
            subject_to_files[sid].append(fpath)
        if skipped > 0:
            print(f"[subject_independent] Warning: skipped {skipped} files without recognizable Subject ID")
        return subject_to_files

    def _get_data_loader_subject_independent(self):
        """
        受试者独立划分（LOSO）：合并全部 chunk 后按 Subject ID 划分，
        保证 train/val/test 受试者互不重叠。要求显式传入 --val_subject / --test_subject
        （由外层 shell 脚本按折循环生成），train = 除 val/test 外的其余全部受试者。
        """
        val_subject = getattr(self.params, 'val_subject', None)
        test_subject = getattr(self.params, 'test_subject', None)
        if not val_subject or not test_subject:
            raise ValueError("split_mode=subject_independent requires both --val_subject and --test_subject")

        all_files = self._collect_all_pickle_files()
        subject_to_files = self._group_files_by_subject(all_files)
        subjects = sorted(subject_to_files.keys())
        if val_subject not in subjects or test_subject not in subjects:
            raise ValueError(f"val_subject={val_subject}, test_subject={test_subject} must be in {subjects}")
        if val_subject == test_subject:
            raise ValueError("val_subject and test_subject must be different")

        train_subjects = [s for s in subjects if s not in (val_subject, test_subject)]

        print("=" * 70)
        print(f"[subject_independent] All subjects ({len(subjects)}): {subjects}")
        print(f"  Train ({len(train_subjects)}): {train_subjects}")
        print(f"  Val   (1): {val_subject}")
        print(f"  Test  (1): {test_subject}")
        print("=" * 70)

        def gather(subj_list):
            files = []
            for s in subj_list:
                files.extend(subject_to_files[s])
            return files

        train_files = gather(train_subjects)
        val_files = gather([val_subject])
        test_files = gather([test_subject])

        train_set = CustomDataset(
            self.datasets_dir, mode='train',
            channel_size=self.channel_size, window_size=self.window_size,
            file_list=train_files,
        )
        val_set = CustomDataset(
            self.datasets_dir, mode='val',
            channel_size=self.channel_size, window_size=self.window_size,
            file_list=val_files,
        )
        test_set = CustomDataset(
            self.datasets_dir, mode='test',
            channel_size=self.channel_size, window_size=self.window_size,
            file_list=test_files,
            return_sample_id=True,
        )
        print(f"[split_mode=subject_independent] Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        return self._build_loaders(train_set, val_set, test_set)

    def _build_loaders(self, train_set, val_set, test_set):
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
