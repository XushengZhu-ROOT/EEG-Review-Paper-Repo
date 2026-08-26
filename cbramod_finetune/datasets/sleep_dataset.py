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
            channel_size=6,
            window_size=30,
            return_sample_id=False,
    ):
        super(CustomDataset, self).__init__()
        # [Sleep] 按最佳 epoch checkpoint 做事后干净推理时用；不影响现有训练/评估
        # 行为（默认 False）。sample_id 就是 preprocess_sleep.py 存盘时用的
        # epoch_id（形如 "sub01_ep0000"，即文件名去掉 .pickle），直接从文件名取，
        # 不用像 Motor 那样重新编号。
        self.return_sample_id = return_sample_id
        mode_dir = os.path.join(data_dir, mode)
        if not os.path.exists(mode_dir):
            raise ValueError(f"Data directory {mode_dir} does not exist!")
        
        # 获取所有pickle文件
        all_files = [os.path.join(mode_dir, file) for file in os.listdir(mode_dir) if file.endswith('.pickle')]
        
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
        
        self.channel_size = channel_size
        self.window_size = window_size
        
        if filtered_count > 0:
            print(f"[{mode}] Filtered out {filtered_count} files with channel size != {channel_size}")
        print(f"[{mode}] Loaded {len(self.files)} files with {channel_size} channels")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        
        # 数据键名：使用'signal'
        data = data_dict['signal']  # shape: (6, 6000)
        
        # 标签：已经是0-4，不需要转换
        label = data_dict['label']  # 0, 1, 2, 3, 4
        
        # 验证通道数
        if data.shape[0] != self.channel_size:
            raise ValueError(f"Expected {self.channel_size} channels, but got {data.shape[0]}. This should not happen after filtering in __init__.")
        
        # Reshape数据：从(6, 6000)转为(6, window_size, 200)
        # 30秒数据，6000个采样点 = 30秒 * 200Hz = 30个窗口 * 200个采样点/窗口
        expected_samples = self.channel_size * self.window_size * 200
        actual_samples = data.size
        
        if actual_samples != expected_samples:
            raise ValueError(
                f"Data size mismatch! Expected {expected_samples} elements "
                f"(channels={self.channel_size}, windows={self.window_size}, samples_per_window=200), "
                f"but got {actual_samples} elements. Data shape: {data.shape}"
            )
        
        data = data.reshape(self.channel_size, self.window_size, 200)

        # 数据归一化：参考其他数据集，除以100
        # 注意：如果数据已经归一化，可能需要调整
        if self.return_sample_id:
            sample_id = os.path.splitext(os.path.basename(file))[0]
            return data / 100.0, label, sample_id
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
