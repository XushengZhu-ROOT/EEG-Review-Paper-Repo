"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

from torch.utils.data import Dataset
from pathlib import Path
import h5py
import bisect
import torch
from einops import rearrange
import tiktoken
import numpy as np
import pickle
import os
import re
from collections import defaultdict
from dataset import standard_1020


def _parse_subject_id(fname):
    """从文件名解析被试编号，例如 'Sub01_8fast_epoch002.pickle' -> 1。
    兼容 'Sub01' / 'Sub1' / 'subject_1' / 'subject01' 等写法。解析失败返回 -1。"""
    base = os.path.basename(fname)
    m = re.search(r'[Ss]ub(?:ject)?_?0*(\d+)', base)
    if m:
        return int(m.group(1))
    return -1


def _build_stable_sample_ids(files, session_of=None):
    """为一批文件构造稳定、可复现、与 shuffle/batch/worker 无关的 sample_id。

    - 无 session:   S{subject:02d}_ep{index:05d}
    - 有 session:   S{subject:02d}_sess{session:02d}_ep{index:05d}

    index 在 (subject[, session]) 分组内，按 basename 排序后从 0 递增。
    因为在 leave-one-subject-out 里同一 subject 的所有文件都在同一个 split，
    组内枚举等价于对该 subject 全量文件枚举，故不同模型、不同 batch 设置下集合完全一致。

    返回: (sample_ids, subject_ids) 两个与 files 等长、顺序对齐的 list。
    """
    subject_ids = [_parse_subject_id(f) for f in files]
    groups = defaultdict(list)
    for i, f in enumerate(files):
        if session_of is not None:
            key = (subject_ids[i], session_of[i])
        else:
            key = (subject_ids[i],)
        groups[key].append(i)

    sample_ids = [None] * len(files)
    for key, idxs in groups.items():
        idxs_sorted = sorted(idxs, key=lambda i: os.path.basename(files[i]))
        for ep, i in enumerate(idxs_sorted):
            subj = key[0]
            if session_of is not None:
                sess = key[1]
                sample_ids[i] = f"S{subj:02d}_sess{sess:02d}_ep{ep:05d}"
            else:
                sample_ids[i] = f"S{subj:02d}_ep{ep:05d}"

    assert len(set(sample_ids)) == len(sample_ids), \
        "sample_id 出现重复，请检查文件名或被试/ session 解析逻辑"
    return sample_ids, subject_ids


def get_chans(ch_names):
    chans = []
    for ch_name in ch_names:
        chans.append(standard_1020.index(ch_name))
    return chans

class KaggleERNLoader(Dataset):
    # increase: 1
    # normal: 0
    def __init__(self, root, files, chan_size, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.chan_size = chan_size
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        self.ch_names = ['FP1', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POZ', 'O1', 'O2']

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Does this segment contain an error response?? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Does this segment contain an error response?? Answer: No <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Does this segment contain an error response?? Answer:'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y = sample["label"]

        # data = torch.FloatTensor(X / 100)
        data = torch.FloatTensor(X)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()


class CustomStressLoader(Dataset):
    # increase: 1
    # normal: 0
    def __init__(self, root, files, chan_size, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.chan_size = chan_size
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        subset_channels_11 = ['FP1','FP2','F3','FZ','F4','FC3','FCZ','FC4','C3','CZ','C4'] # 11 channels
        subset_channels_20 =['FP1','F7','F3','F8','FZ','FC4', 'FT8', 'T3', 'C3', 'CZ', 'T4', 'TP7', 'CP3', 'CPZ', 'CP4','T5', 'P3', 'PZ', 'P4', 'T6'] # 20 channels
        subset_channels_30 = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']
        
        if self.chan_size == 11:
            self.ch_names = subset_channels_11
        elif self.chan_size == 20:
            self.ch_names = subset_channels_20
        elif self.chan_size == 30:
            self.ch_names = subset_channels_30
        else:
            self.ch_names = []
            raise ValueError(f"Undefined channel size: {self.chan_size}")

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Does this EEG segment indicate increased stress? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Does this EEG segment indicate increased stress? Answer: No <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Does this EEG segment indicate increased stress? Answer:'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = sample["y"]

        # data = torch.FloatTensor(X / 100)
        data = torch.FloatTensor(X)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
    



    


 
    




 




class SEED7Loader(Dataset):
    """SEED 6-class emotion classification loader (happy, sad, disgust, fear, surprise, anger) - neutral excluded"""
    def __init__(self, root, files, chan_size, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.chan_size = chan_size
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        # Filter out neutral (label=2) samples
        self.files = []
        for file in files:
            try:
                sample = pickle.load(open(os.path.join(root, file), "rb"))
                label = sample["label"]
                # Only keep samples that are not neutral (label != 2)
                if label != 2:
                    self.files.append(file)
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
                continue
        
        # Label mapping: original -> new (removing neutral=2)
        # 0->0 (happy), 1->1 (sad), 3->2 (disgust), 4->3 (fear), 5->4 (surprise), 6->5 (anger)
        self.label_map = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5}

        # 62 channels for SEED dataset
        self.ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            # 6 emotion classes: happy(0), sad(1), disgust(2), fear(3), surprise(4), anger(5) - neutral excluded
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which emotion does this EEG segment express? Options: (A) happy. (B) sad. (C) disgust. (D) fear. (E) surprise. (F) anger. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which emotion does this EEG segment express? Options: (A) happy. (B) sad. (C) disgust. (D) fear. (E) surprise. (F) anger. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which emotion does this EEG segment express? Options: (A) happy. (B) sad. (C) disgust. (D) fear. (E) surprise. (F) anger. Answer: (C) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which emotion does this EEG segment express? Options: (A) happy. (B) sad. (C) disgust. (D) fear. (E) surprise. (F) anger. Answer: (D) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which emotion does this EEG segment express? Options: (A) happy. (B) sad. (C) disgust. (D) fear. (E) surprise. (F) anger. Answer: (E) <|endoftext|>')),
                5: torch.IntTensor([50257] + encode('Question: Which emotion does this EEG segment express? Options: (A) happy. (B) sad. (C) disgust. (D) fear. (E) surprise. (F) anger. Answer: (F) <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which emotion does this EEG segment express? Options: (A) happy. (B) sad. (C) disgust. (D) fear. (E) surprise. (F) anger. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y_original = sample["label"]
        
        # Map label: remove neutral (2) and remap others
        Y = self.label_map[Y_original]

        data = torch.FloatTensor(X / 100)
        # data = torch.FloatTensor(X)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            # 返回文件名（epoch_id）：去掉.pickle后缀
            epoch_id = os.path.splitext(self.files[index])[0]
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), epoch_id
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()

 

class MotorLoader(Dataset):
    """Motor imagery 6-class classification loader (Label0, Walk, 8, Horizontal, Vertical, Pick)

    root 可以是一个目录（files 为相对文件名），也可以为 None（files 为绝对路径，
    用于 leave-one-subject-out：同一 split 的文件可能来自 train/val/test 多个物理目录）。
    """
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        # 稳定的 sample_id / subject_id（构造时确定，随 __getitem__ 返回，不受 shuffle 影响）
        # Motor 数据集无 session，故 sample_id 形如 S{subject:02d}_ep{index:05d}
        self.sample_ids, self.subject_ids = _build_stable_sample_ids(files, session_of=None)

        # 20 channels for Motor dataset
        self.ch_names = ['F7','FP1','FP2','F8','F3','FZ','F4','C3','CZ','P8','P7','PZ','P4','T3','P3','O1','O2','C4','T4','A2']

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            # 6 motor classes: Label0(0), Walk(1), 8(2), Horizontal(3), Vertical(4), Pick(5)
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which motor imagery type does this EEG segment belong to? Options: (A) Label0. (B) Walk. (C) 8. (D) Horizontal. (E) Vertical. (F) Pick. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which motor imagery type does this EEG segment belong to? Options: (A) Label0. (B) Walk. (C) 8. (D) Horizontal. (E) Vertical. (F) Pick. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which motor imagery type does this EEG segment belong to? Options: (A) Label0. (B) Walk. (C) 8. (D) Horizontal. (E) Vertical. (F) Pick. Answer: (C) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which motor imagery type does this EEG segment belong to? Options: (A) Label0. (B) Walk. (C) 8. (D) Horizontal. (E) Vertical. (F) Pick. Answer: (D) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which motor imagery type does this EEG segment belong to? Options: (A) Label0. (B) Walk. (C) 8. (D) Horizontal. (E) Vertical. (F) Pick. Answer: (E) <|endoftext|>')),
                5: torch.IntTensor([50257] + encode('Question: Which motor imagery type does this EEG segment belong to? Options: (A) Label0. (B) Walk. (C) 8. (D) Horizontal. (E) Vertical. (F) Pick. Answer: (F) <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which motor imagery type does this EEG segment belong to? Options: (A) Label0. (B) Walk. (C) 8. (D) Horizontal. (E) Vertical. (F) Pick. Answer: ('))

    def __len__(self):
        return len(self.files)

    def _resolve_path(self, index):
        f = self.files[index]
        if self.root is None or os.path.isabs(f):
            return f
        return os.path.join(self.root, f)

    def __getitem__(self, index):
        sample = pickle.load(open(self._resolve_path(index), "rb"))
        X = sample["signal"]
        Y = sample["label"]

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            # 评估阶段额外返回稳定的 sample_id 与 subject_id，供事后计算所有下游指标
            return (X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(),
                    self.sample_ids[index], self.subject_ids[index])
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        

class SleepLoader(Dataset):
    """Sleep stage 5-class classification loader (0, 1, 2, 3, 4)
    
    Data specifications:
    - 6 channels: ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
    - 5 classes: 0, 1, 2, 3, 4
    - Data length: 30 seconds (6000 samples at 200Hz sampling rate)
    """
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        # 6 channels for Sleep dataset: ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
        self.ch_names = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            # 5 sleep stage classes: 0, 1, 2, 3, 4
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which sleep stage does this EEG segment belong to? Options: (A) Stage 0. (B) Stage 1. (C) Stage 2. (D) Stage 3. (E) Stage 4. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which sleep stage does this EEG segment belong to? Options: (A) Stage 0. (B) Stage 1. (C) Stage 2. (D) Stage 3. (E) Stage 4. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which sleep stage does this EEG segment belong to? Options: (A) Stage 0. (B) Stage 1. (C) Stage 2. (D) Stage 3. (E) Stage 4. Answer: (C) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which sleep stage does this EEG segment belong to? Options: (A) Stage 0. (B) Stage 1. (C) Stage 2. (D) Stage 3. (E) Stage 4. Answer: (D) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which sleep stage does this EEG segment belong to? Options: (A) Stage 0. (B) Stage 1. (C) Stage 2. (D) Stage 3. (E) Stage 4. Answer: (E) <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which sleep stage does this EEG segment belong to? Options: (A) Stage 0. (B) Stage 1. (C) Stage 2. (D) Stage 3. (E) Stage 4. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y = int(sample["label"])

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()