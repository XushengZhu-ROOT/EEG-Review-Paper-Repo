import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import pickle


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            channel_size=62,
            window_size=5,
    ):
        super(CustomDataset, self).__init__()
        mode_dir = os.path.join(data_dir, mode)
        if not os.path.exists(mode_dir):
            raise ValueError(f"Data directory {mode_dir} does not exist!")
        
        # 获取所有子文件夹中的pickle文件
        self.files = []
        for subject_dir in os.listdir(mode_dir):
            subject_path = os.path.join(mode_dir, subject_dir)
            if os.path.isdir(subject_path):
                for file in os.listdir(subject_path):
                    if file.endswith('.pickle'):
                        self.files.append(os.path.join(subject_path, file))
        
        self.channel_size = channel_size
        self.window_size = window_size
        print(f"[{mode}] Loaded {len(self.files)} files with {channel_size} channels")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['signal']  # shape: (62, 1000)
        label = data_dict['label']  # label: 0-6 (7分类)
        
        # 验证通道数
        if data.shape[0] != self.channel_size:
            raise ValueError(f"Expected {self.channel_size} channels, but got {data.shape[0]} in file {file}")
        
        # Reshape数据：从(62, 1000)转为(62, window_size, 200)
        # 1000个采样点 = 5秒 * 200Hz = 5个窗口 * 200个采样点/窗口
        if data.shape[1] != self.window_size * 200:
            raise ValueError(
                f"Expected {self.window_size * 200} samples (for {self.window_size} windows), "
                f"but got {data.shape[1]} in file {file}"
            )
        
        data = data.reshape(self.channel_size, self.window_size, 200)
        
        # 数据归一化：除以100（参考kaggleern和motortask）
        return data / 100.0, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), torch.from_numpy(y_label).long()


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

