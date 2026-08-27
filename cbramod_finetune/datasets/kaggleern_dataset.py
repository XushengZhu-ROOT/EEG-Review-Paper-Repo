import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from scipy import signal

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            channel_size=56,
            window_size=3,
            return_sample_id=False,
    ):
        super(CustomDataset, self).__init__()
        # [KaggleERN bestval] 按最佳 epoch checkpoint 做事后干净推理时用；不影响现有
        # 训练/评估行为（默认 False）。sample_id 就是 preprocess_KaggleERN_new.ipynb
        # 存盘时用的 epoch_id（形如 "S02_Sess01_FB004"，即文件名去掉 .pickle），
        # 前缀 "S(\d+)_" 就是被试号，跟 finetune_evaluator.py 的
        # save_fold_predictions_npz 已经在用的 subj_re 保持一致。
        self.return_sample_id = return_sample_id
        # [bugfix] os.listdir 不保证顺序；sort 一下让 train/val/test 每次构造的文件
        # 顺序都一样，方便复现和跟 sample_id 排序后的 npz 对齐（val/test 用 shuffle=False，
        # 顺序不稳定会让"这次跑出来的第 N 个 batch"不可复现，虽不影响最终聚合指标）。
        self.files = sorted(
            os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))
        )
        self.channel_size = channel_size
        self.window_size = window_size

    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['signal']
        # print("Shape of data[X]: ", data.shape)
        label = data_dict['label']
        # data = signal.resample(data, 2000, axis=-1)
        data = data.reshape(self.channel_size, self.window_size, 200) # for 30 channels
        if self.return_sample_id:
            sample_id = os.path.splitext(os.path.basename(file))[0]
            return data/100, label, sample_id
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.channel_size = params.channel_size
        self.window_size = params.window_size

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train', channel_size=self.channel_size, window_size=self.window_size)
        val_set = CustomDataset(self.datasets_dir, mode='val', channel_size=self.channel_size, window_size=self.window_size)
        test_set = CustomDataset(self.datasets_dir, mode='test', channel_size=self.channel_size, window_size=self.window_size)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
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
