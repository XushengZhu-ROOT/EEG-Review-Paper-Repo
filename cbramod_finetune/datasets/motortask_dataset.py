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
    """从文件名或 epoch_id 中提取受试者 ID，例如 Sub01_8fast_epoch002 -> Sub01"""
    basename = os.path.basename(name)
    match = re.match(r'(Sub\d+)_', basename)
    if match:
        return match.group(1)
    match = re.search(r'(Sub\d+)', basename)
    if match:
        return match.group(1)
    return None


# ===== [R2 新增] 跨模型稳定的 sample_id =====
# epoch_id 形如 "Sub04_Walkslow_epoch009"（datamake/motiondata_preprocess.ipynb 生成）。
# 任务 token 的基础任务名固定为以下六种（对应 LABEL_MAP 里的六类），速度后缀固定为三种。
# 用固定 offset（而不是对某个受试者全部文件排序后的“第几个”这种全局流水号）算 index，
# 是为了避免某个模型在某一个 task 边界处恰好多切/少切 1 个 epoch 时，
# 导致该受试者后面所有 epoch 的编号跟着整体错位——用固定 offset，
# 每个 epoch 的 sample_id 只取决于它自己的 (subject, base_task, speed, task 内原始序号)，
# 不受同一受试者其他任务实际切出多少个 epoch 影响。
_SAMPLE_ID_TASK_ORDER = ['Walk', '8', 'Horizontal', 'Vertical', 'Pick', 'Stair']
_SAMPLE_ID_SPEED_ORDER = ['slow', 'medium', 'fast']
_SAMPLE_ID_TASK_OFFSET = 3000
_SAMPLE_ID_SPEED_OFFSET = 1000
_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(.+?)_epoch(\d+)$')


def _parse_task_token(task_token):
    for speed in _SAMPLE_ID_SPEED_ORDER:
        if task_token.endswith(speed):
            return task_token[: -len(speed)], speed
    raise ValueError(f"Cannot parse speed suffix (slow/medium/fast) from task token: {task_token!r}")


def compute_sample_id(epoch_id):
    """
    由 epoch_id（如 'Sub04_Walkslow_epoch009'）确定性地生成 sample_id，
    格式：S{subject:02d}_ep{index:05d}（MotorTask 无 session 概念，不加 sess 段）。
    纯函数，只依赖 epoch_id 字符串本身，在数据集构造（列文件）时即可算出，
    与 shuffle / batch_size / num_workers 无关；对同一批底层 epoch 文件，
    任何模型跑出来算出的 sample_id 都应一致。
    """
    m = _SAMPLE_ID_RE.match(epoch_id)
    if not m:
        raise ValueError(f"Cannot parse epoch_id for sample_id: {epoch_id!r}")
    subject_num = int(m.group(1))
    task_token = m.group(2)
    local_idx = int(m.group(3))
    base_task, speed = _parse_task_token(task_token)
    if base_task not in _SAMPLE_ID_TASK_ORDER:
        raise ValueError(
            f"Unknown base task {base_task!r} parsed from epoch_id {epoch_id!r}; "
            f"expected one of {_SAMPLE_ID_TASK_ORDER}"
        )
    task_idx = _SAMPLE_ID_TASK_ORDER.index(base_task)
    speed_idx = _SAMPLE_ID_SPEED_ORDER.index(speed)
    global_index = task_idx * _SAMPLE_ID_TASK_OFFSET + speed_idx * _SAMPLE_ID_SPEED_OFFSET + local_idx
    if global_index > 99999:
        raise ValueError(f"sample_id index overflow (>99999) for epoch_id {epoch_id!r}: {global_index}")
    return f"S{subject_num:02d}_ep{global_index:05d}"


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            channel_size=20,
            window_size=1,
            # ===== [R1 新增] 可选：直接传入文件列表（用于受试者独立划分）=====
            # 若 file_list 不为 None，则忽略 data_dir/mode 目录扫描，直接使用该列表。
            # 旧的 random-epoch 划分路径不会传 file_list，行为与修改前完全一致。
            file_list=None,
    ):
        super(CustomDataset, self).__init__()
        self.channel_size = channel_size
        self.window_size = window_size

        # ===== [R1 新增] 受试者独立划分：使用预先筛选好的 file_list =====
        if file_list is not None:
            all_files = list(file_list)
            mode_tag = mode
        else:
            # ===== [保留原逻辑] 按 train/val/test 子目录加载（随机 epoch 划分后的数据）=====
            mode_dir = os.path.join(data_dir, mode)
            if not os.path.exists(mode_dir):
                raise ValueError(f"Data directory {mode_dir} does not exist!")
            all_files = [os.path.join(mode_dir, file) for file in os.listdir(mode_dir) if file.endswith('.pickle')]
            mode_tag = mode

        # 过滤：只保留通道数为channel_size的文件
        self.files = []
        filtered_count = 0
        for file in all_files:
            try:
                data_dict = pickle.load(open(file, 'rb'))
                data = data_dict['signal']
                if data.shape[0] == channel_size:
                    self.files.append(file)
                else:
                    filtered_count += 1
            except Exception as e:
                # 如果文件读取失败，跳过该文件
                print(f"Warning: Failed to read {file}: {e}")
                filtered_count += 1

        if filtered_count > 0:
            print(f"[{mode_tag}] Filtered out {filtered_count} files with channel size != {channel_size}")
        print(f"[{mode_tag}] Loaded {len(self.files)} files with {channel_size} channels")

        # ===== [R2 新增] sample_id 在数据集构造（列文件）时一次性确定，
        # 与 self.files 一一对应；不依赖 shuffle / batch_size / num_workers。
        # 解析失败直接报错退出（不静默跳过），避免后面 npz 里混进坏数据。
        self.sample_ids = [
            compute_sample_id(os.path.splitext(os.path.basename(f))[0]) for f in self.files
        ]
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(f"[{mode_tag}] Duplicate sample_id detected after computing sample_ids; "
                              f"check for duplicate/conflicting epoch files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        sample_id = self.sample_ids[idx]
        data_dict = pickle.load(open(file, 'rb'))
        
        # 数据键名：根据notebook，使用'signal'
        data = data_dict['signal']  # shape: (20, 200)
        
        # # 标签：原始标签是1-6，需要转换为0-5（PyTorch CrossEntropyLoss要求从0开始）
        # label = data_dict['label'] - 1  # 将1-6转为0-5
        label = data_dict['label']
        # Reshape数据：从(20, 200)转为(20, window_size, 200)
        # 对于window_size=1: (20, 200) -> (20, 1, 200)
        # 如果数据长度更长，需要相应调整window_size
        # 注意：通道数检查已在__init__中完成，这里应该不会出现不匹配的情况
        if data.shape[0] != self.channel_size:
            raise ValueError(f"Expected {self.channel_size} channels, but got {data.shape[0]}. This should not happen after filtering in __init__.")
        
        expected_samples = self.channel_size * self.window_size * 200
        actual_samples = data.size
        
        if actual_samples != expected_samples:
            # 如果数据长度不匹配，尝试reshape
            if actual_samples == self.channel_size * 200:
                # 数据是(20, 200)，window_size应该是1
                data = data.reshape(self.channel_size, self.window_size, 200)
            else:
                raise ValueError(
                    f"Data size mismatch! Expected {expected_samples} elements "
                    f"(channels={self.channel_size}, windows={self.window_size}, samples_per_window=200), "
                    f"but got {actual_samples} elements. Data shape: {data.shape}"
                )
        else:
            data = data.reshape(self.channel_size, self.window_size, 200)
        
        # 数据归一化：参考kaggleern，除以100
        # 如果后续发现数据已经归一化，可以调整或移除这一步
        epoch_id = data_dict.get('epoch_id', os.path.splitext(os.path.basename(file))[0])
        return data / 100.0, label, epoch_id, sample_id

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        epoch_ids = [x[2] for x in batch]
        sample_ids = [x[3] for x in batch]
        # ===== Debug: 仅在第一次 collate 时打印标签统计，帮助检查越界问题 =====
        if not hasattr(self, "_debug_labels_printed"):
            try:
                y_tensor = torch.from_numpy(y_label).long()
                print("[DEBUG][MotorTask] Batch label stats:")
                print(f"  shape: {y_tensor.shape}, dtype: {y_tensor.dtype}")
                print(f"  min: {y_tensor.min().item()}, max: {y_tensor.max().item()}")
                unique_vals = torch.unique(y_tensor)
                max_to_show = min(50, unique_vals.numel())
                print(f"  unique (first {max_to_show}): {unique_vals.view(-1)[:max_to_show].tolist()}")
            except Exception as e:
                print(f"[DEBUG][MotorTask] Error while printing label stats: {e}")
            self._debug_labels_printed = True
        # ==============================================================
        return to_tensor(x_data), torch.from_numpy(y_label).long(), epoch_ids, sample_ids


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
          - subject_independent（R1 新增）：按受试者严格划分，避免相邻 epoch 泄漏
        """
        # ===== [R1 新增] 受试者独立划分开关 =====
        split_mode = getattr(self.params, 'split_mode', 'random_epoch')
        if split_mode == 'subject_independent':
            return self._get_data_loader_subject_independent()

        # ===== [保留原逻辑] 随机 epoch 划分（读现成的 train/val/test 目录）=====
        return self._get_data_loader_random_epoch()

    def _get_data_loader_random_epoch(self):
        """[旧方法] 使用预先按 epoch 随机划分好的 train/val/test 目录。"""
        train_set = CustomDataset(
            self.datasets_dir,
            mode='train',
            channel_size=self.channel_size,
            window_size=self.window_size
        )
        val_set = CustomDataset(
            self.datasets_dir,
            mode='val',
            channel_size=self.channel_size,
            window_size=self.window_size
        )
        test_set = CustomDataset(
            self.datasets_dir,
            mode='test',
            channel_size=self.channel_size,
            window_size=self.window_size
        )
        print(f"[split_mode=random_epoch] Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        print(f"Total: {len(train_set) + len(val_set) + len(test_set)}")
        return self._build_loaders(train_set, val_set, test_set)

    def _collect_all_pickle_files(self):
        """
        [R1 新增] 从 train/val/test 三个子目录收集全部 pickle。
        现有磁盘布局是按 epoch 随机划进三个目录的，但每个受试者都出现在三个目录中；
        受试者独立评估时需要先合并再按 Subject ID 重新划分。
        """
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
        """[R1 新增] 按受试者 ID 分组文件路径。"""
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

    def _make_subject_fold_split(self, subjects):
        """
        [R1 新增] 构造一折受试者划分：18 train / 1 val / 1 test。

        single_fold_debug=True 时只跑第一折：
          train = subjects[:-2], val = subjects[-2], test = subjects[-1]
        例如排序后末尾为 Sub20 / Sub21，则 val=Sub20, test=Sub21。

        后续若要做完整 LOSO，可在此扩展多折循环；当前仅实现 dry-run 单折。
        """
        subjects = sorted(subjects)
        n = len(subjects)
        if n < 3:
            raise ValueError(f"Need at least 3 subjects for subject-independent split, got {n}: {subjects}")

        single_fold_debug = getattr(self.params, 'single_fold_debug', True)
        # 可选：通过 CLI 手动指定 val/test 受试者（便于复现/对比）
        val_subject = getattr(self.params, 'val_subject', None)
        test_subject = getattr(self.params, 'test_subject', None)
        # ===== [R2 新增] 无验证子集的经典 LOSO：N-1 train / 1 test，不划 val =====
        # 仅新增分支，不改动下面 val_subject/test_subject 与默认单折逻辑，
        # 保证 R1（18-1-1）的行为和结果完全不受影响。
        no_val_subject = getattr(self.params, 'no_val_subject', False)

        if no_val_subject:
            if not test_subject:
                raise ValueError("no_val_subject=True requires --test_subject to be set")
            if test_subject not in subjects:
                raise ValueError(f"test_subject={test_subject} must be in {subjects}")
            train_subjects = [s for s in subjects if s != test_subject]
            val_subjects = []
            test_subjects = [test_subject]
            fold_id = 'no_val'
        elif val_subject and test_subject:
            if val_subject not in subjects or test_subject not in subjects:
                raise ValueError(
                    f"val_subject={val_subject}, test_subject={test_subject} must be in {subjects}"
                )
            if val_subject == test_subject:
                raise ValueError("val_subject and test_subject must be different")
            train_subjects = [s for s in subjects if s not in (val_subject, test_subject)]
            val_subjects = [val_subject]
            test_subjects = [test_subject]
            fold_id = 'custom'
        else:
            # 默认第一折：最后两个受试者分别做 val / test，其余全部进 train
            # （对应 prompt 中的 single_fold_debug dry run）
            if not single_fold_debug:
                print("[subject_independent] single_fold_debug=False，但完整 LOSO 尚未实现；仍只运行第一折。")
            train_subjects = subjects[:-2]
            val_subjects = [subjects[-2]]
            test_subjects = [subjects[-1]]
            fold_id = 0

        print("=" * 70)
        print(f"[subject_independent] fold={fold_id}, single_fold_debug={single_fold_debug}")
        print(f"  All subjects ({n}): {subjects}")
        print(f"  Train ({len(train_subjects)}): {train_subjects}")
        print(f"  Val   ({len(val_subjects)}): {val_subjects}")
        print(f"  Test  ({len(test_subjects)}): {test_subjects}")
        print("=" * 70)
        return train_subjects, val_subjects, test_subjects

    def _get_data_loader_subject_independent(self):
        """
        [R1 新增] Subject-Independent 划分：
        合并全部 epoch 后按 Subject ID 划分，保证 train/val/test 受试者互不重叠。
        """
        all_files = self._collect_all_pickle_files()
        subject_to_files = self._group_files_by_subject(all_files)
        subjects = sorted(subject_to_files.keys())
        print(f"[subject_independent] Found {len(subjects)} subjects, {len(all_files)} pickle files total")

        train_subjects, val_subjects, test_subjects = self._make_subject_fold_split(subjects)

        def gather(subj_list):
            files = []
            for s in subj_list:
                files.extend(subject_to_files[s])
            return files

        train_files = gather(train_subjects)
        val_files = gather(val_subjects)
        test_files = gather(test_subjects)

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
        )
        print(f"[split_mode=subject_independent] Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        print(f"Total: {len(train_set) + len(val_set) + len(test_set)}")
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
