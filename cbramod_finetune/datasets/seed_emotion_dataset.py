import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import pickle
import re


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            channel_size=62,
            window_size=4,
    ):
        super(CustomDataset, self).__init__()
        mode_dir = os.path.join(data_dir, mode)
        if not os.path.exists(mode_dir):
            raise ValueError(f"Data directory {mode_dir} does not exist!")
        
        # 获取所有子文件夹中的pickle文件，并过滤掉neutral(label=2)的数据
        all_files = []
        for subject_dir in os.listdir(mode_dir):
            subject_path = os.path.join(mode_dir, subject_dir)
            if os.path.isdir(subject_path):
                for file in os.listdir(subject_path):
                    if file.endswith('.pickle'):
                        all_files.append(os.path.join(subject_path, file))
        
        # 过滤掉neutral(label=2)的数据
        self.files = []
        filtered_count = 0
        for file in all_files:
            try:
                data_dict = pickle.load(open(file, 'rb'))
                label = data_dict['label']
                # 只保留非neutral的数据 (label != 2)
                if label != 2:
                    self.files.append(file)
                else:
                    filtered_count += 1
            except Exception as e:
                print(f"Warning: Failed to read {file}: {e}")
                filtered_count += 1
        
        self.channel_size = channel_size
        self.window_size = window_size
        if filtered_count > 0:
            print(f"[{mode}] Filtered out {filtered_count} files with neutral label (label=2)")
        print(f"[{mode}] Loaded {len(self.files)} files with {channel_size} channels (excluding neutral)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['signal']  # shape: (62, 800)
        original_label = data_dict['label']  # 原始label: 0-6 (已过滤掉2)
        
        # 重新映射标签：移除neutral(2)后，将后面的标签前移
        # 原始映射: happy=0, sad=1, neutral=2, disgust=3, fear=4, surprise=5, anger=6
        # 新映射:   happy=0, sad=1, disgust=2, fear=3, surprise=4, anger=5
        # 所以: 0->0, 1->1, 3->2, 4->3, 5->4, 6->5
        if original_label == 2:
            # 理论上不应该出现，因为已经过滤了，但为了安全起见还是处理一下
            raise ValueError(f"Found neutral label (2) in filtered dataset: {file}")
        elif original_label < 2:
            # 0 (happy) -> 0, 1 (sad) -> 1
            label = original_label
        else:
            # 3->2, 4->3, 5->4, 6->5
            label = original_label - 1
        
        # 验证通道数
        if data.shape[0] != self.channel_size:
            raise ValueError(f"Expected {self.channel_size} channels, but got {data.shape[0]} in file {file}")
        
        # Reshape数据：从(62, 800)转为(62, window_size, 200)
        # 800个采样点 = 4秒 * 200Hz = 4个窗口 * 200个采样点/窗口
        if data.shape[1] != self.window_size * 200:
            raise ValueError(
                f"Expected {self.window_size * 200} samples (for {self.window_size} windows), "
                f"but got {data.shape[1]} in file {file}"
            )
        
        data = data.reshape(self.channel_size, self.window_size, 200)
        
        # 从文件名提取epoch_id（去掉路径和.pickle后缀）
        file_basename = os.path.basename(file)
        epoch_id = file_basename.replace('.pickle', '')
        
        # 数据归一化：除以100（参考kaggleern和motortask）
        return data / 100.0, label, epoch_id

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        epoch_ids = [x[2] for x in batch]  # 保留epoch_id用于后续投票评估
        return to_tensor(x_data), torch.from_numpy(y_label).long(), epoch_ids


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.channel_size = params.channel_size
        self.window_size = params.window_size

    def get_data_loader(self):
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
        print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        print(f"Total: {len(train_set) + len(val_set) + len(test_set)}")
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

