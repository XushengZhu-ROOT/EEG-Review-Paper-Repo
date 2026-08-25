import os
import pdb
import torch
import pickle
import numpy as np
import re
from typing import Tuple, List, Dict
from batcher.base import EEGDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from utils import (
    compute_motor_sample_id,
    extract_motor_subject_id,
    compute_stress_sample_id,
    extract_stress_subject_id,
)




class KaggleERNDataset(EEGDataset):
    def __init__(
        self,
        filenames,
        sample_keys,
        chunk_len=500,
        num_chunks=10,
        ovlp=50,
        root_path="",
        matrix_p_path=None,
        gpt_only=True,
        # [LOSO] 是否在 __getitem__ 中额外返回 sample_id（仅 Stress 任务 subject_independent
        # 划分使用）。默认 False，与现有 KaggleERN / stress random_epoch 调用点行为完全一致。
        # Stress 的 chunk 文件名（'SubNN_increase_edfNN_chunkNNNN'）才能被
        # compute_stress_sample_id/extract_stress_subject_id 解析；KaggleERN 不支持置 True。
        return_sample_id=False,
    ):
        super().__init__(
            filenames,
            sample_keys,
            chunk_len,
            num_chunks,
            ovlp,
            root_path=root_path,
            gpt_only=gpt_only,
        )

        self.return_sample_id = return_sample_id

        # Load the channel transformation matrix P
        # The P matrix should be (22, N_your_channels)
        try:
            self.P = np.load(matrix_p_path)
            print(f"Loaded P matrix for channel mapping with shape: {self.P.shape}")
            if self.P.shape[0] != 22:
                raise ValueError(
                    f"P matrix output channels must be 22, but got {self.P.shape[0]}."
                )
        except FileNotFoundError:
            print(
                f"ERROR: P matrix not found at {matrix_p_path}. Cannot perform channel mapping."
            )
            self.P = None

        self.trials, self.labels, self.sample_ids, self.subject_ids, self.num_trials_per_file = self.get_trials_all()

    def __len__(self):
        return self.num_trials_per_file

    def map2pret(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the channel transformation P: Data_mapped = P @ Data_original
        data shape: (N_your_channels, N_samples)
        """
        if self.P is None:
            raise RuntimeError("P matrix is not loaded. Cannot map channels.")

        # Check for channel count consistency
        if self.P.shape[1] != data.shape[0]:
            raise ValueError(
                f"Channel mismatch! P matrix expects {self.P.shape[1]} input channels, "
                f"but the loaded data has {data.shape[0]} channels."
            )

        # matrix multiplication
        return np.matmul(self.P, data)  # Output shape: (22, N_samples)

    def get_trials_all(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], int]:
        trials_all = []
        labels_all = []
        sample_ids_all = []
        subject_ids_all = []
        total_num = 0

        for file_path in self.filenames:
            try:
                with open(file_path, "rb") as f:
                    sample = pickle.load(f)
            except Exception as e:
                print(f"❌ 載入檔案 {file_path} 時發生錯誤: {e}")
                continue

            # 試次數據 shape: (N_your_channels, N_samples)
            if "signal" in sample:
                trial_data = sample["signal"]
            elif "X" in sample:
                trial_data = sample["X"]
            else:
                raise KeyError("Sample data must contain either 'signal' or 'X' key.")

            if "label" in sample:
                label = sample["label"]
            elif "y" in sample:
                label = sample["y"]
            else:
                raise KeyError("Sample data must contain either 'label' or 'y' key.")

            mapped_data = self.map2pret(trial_data)

            trials_all.append(mapped_data)
            labels_all.append(label)

            if self.return_sample_id:
                chunk_id = os.path.splitext(os.path.basename(file_path))[0]
                sample_ids_all.append(compute_stress_sample_id(chunk_id))
                subj = extract_stress_subject_id(file_path)
                if subj is None:
                    raise ValueError(f"Cannot extract subject id from {file_path}")
                subject_ids_all.append(int(subj[3:]))

        total_num = len(trials_all)

        if self.return_sample_id and len(set(sample_ids_all)) != len(sample_ids_all):
            raise ValueError(
                "Duplicate sample_id detected in KaggleERNDataset; "
                "check for duplicate/conflicting chunk files."
            )

        # (Total_Trials, channel_size, chunk_len*sample_rate)
        trials_all_arr = np.stack(trials_all, axis=0)

        return (
            self.normalize(trials_all_arr),
            np.array(labels_all).flatten(),
            sample_ids_all,
            subject_ids_all,
            total_num,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads pickle file, maps channels, and processes the sample.
        """
        trial_data_normalized_once = self.trials[idx]
        label = self.labels[idx]

        result = self.preprocess_sample(
            sample=trial_data_normalized_once,
            seq_len=self.num_chunks,  # self.num_chunks is the 'seq_len' argument in preprocess_sample
            labels=label,  # Pass label as a numpy array for consistent handling
        )
        if self.return_sample_id:
            result["sample_id"] = self.sample_ids[idx]
        return result


class Motor6ClassDataset(EEGDataset):
    """
    运动6分类数据集类
    - 20通道输入，转换为22通道
    - 6分类任务 (0: Walk, 1: 8, 2: Horizontal, 3: Vertical, 4: Pick, 5: Stair)
    - 1秒数据（250个时间点@250Hz），复制成2秒（500个时间点）以匹配预训练模型
    """
    def __init__(
        self,
        filenames,
        sample_keys,
        chunk_len=250,  # 1秒数据，用于分成2个chunks
        num_chunks=2,
        ovlp=0,  # 不重叠
        root_path="",
        matrix_p_path=None,
        gpt_only=True,
        return_sample_id=False,
        normalization=True,
    ):
        super().__init__(
            filenames,
            sample_keys,
            chunk_len,
            num_chunks,
            ovlp,
            root_path=root_path,
            gpt_only=gpt_only,
            normalization=normalization,
        )

        # [LOSO] 是否在 __getitem__ 中额外返回 sample_id；默认 False，
        # 与现有非 LOSO 调用方行为完全一致。
        self.return_sample_id = return_sample_id

        # 加载通道转换矩阵 P (22, 20)
        try:
            self.P = np.load(matrix_p_path)
            print(f"Loaded P matrix for channel mapping with shape: {self.P.shape}")
            if self.P.shape[0] != 22:
                raise ValueError(
                    f"P matrix output channels must be 22, but got {self.P.shape[0]}."
                )
            if self.P.shape[1] != 20:
                raise ValueError(
                    f"P matrix input channels must be 20, but got {self.P.shape[1]}."
                )
        except FileNotFoundError:
            print(
                f"ERROR: P matrix not found at {matrix_p_path}. Cannot perform channel mapping."
            )
            raise

        self.trials, self.labels, self.sample_ids, self.subject_ids, self.num_trials_per_file = self.get_trials_all()
        print(f"✓ 成功加载 {self.num_trials_per_file} 个样本")

    def __len__(self):
        return self.num_trials_per_file

    def map2pret(self, data: np.ndarray) -> np.ndarray:
        """
        将20通道数据映射到22通道
        data shape: (20, N_samples) → (22, N_samples)
        """
        if self.P is None:
            raise RuntimeError("P matrix is not loaded. Cannot map channels.")

        if self.P.shape[1] != data.shape[0]:
            raise ValueError(
                f"Channel mismatch! P matrix expects {self.P.shape[1]} input channels, "
                f"but the loaded data has {data.shape[0]} channels."
            )

        return np.matmul(self.P, data)  # Output shape: (22, N_samples)

    def get_trials_all(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], int]:
        """
        加载所有数据并映射通道
        返回: (trials, labels, sample_ids, subject_ids, num_trials)
        - 自动跳过30通道的数据
        - 将1秒数据（250个时间点）复制成2秒（500个时间点）
        - sample_ids/subject_ids 只在 self.return_sample_id=True 时才计算（[LOSO]），
          与 trials_all/labels_all 在同一个循环里同步 append，保证跳过的样本不会错位。
        """
        trials_all = []
        labels_all = []
        sample_ids_all = []
        subject_ids_all = []
        skipped_30ch = 0
        skipped_other = 0

        for filename in self.filenames:
            file_path = os.path.join(self.root_path, filename)
            try:
                with open(file_path, "rb") as f:
                    sample = pickle.load(f)
            except Exception as e:
                skipped_other += 1
                continue

            # 提取数据
            if "signal" in sample:
                trial_data = sample["signal"]  # shape: (channels, time)
            else:
                skipped_other += 1
                continue

            # 检查通道数：跳过30通道的数据
            if trial_data.shape[0] == 30:
                skipped_30ch += 1
                continue

            # 检查是否为20通道
            if trial_data.shape[0] != 20:
                skipped_other += 1
                continue

            # 检查时间长度：应该是250个时间点（1秒@250Hz）
            if trial_data.shape[1] != 250:
                skipped_other += 1
                continue

            if "label" in sample:
                label = sample["label"]
            else:
                skipped_other += 1
                continue

            # 将1秒数据（250个时间点）复制成2秒（500个时间点）
            # 这样可以使用chunk_len=250，与预训练模型匹配
            trial_data = np.tile(trial_data, (1, 2))  # shape: (20, 500)

            # 映射到22通道
            mapped_data = self.map2pret(trial_data)  # (22, 500)

            trials_all.append(mapped_data)
            labels_all.append(label)

            if self.return_sample_id:
                epoch_id = os.path.splitext(os.path.basename(file_path))[0]
                sample_ids_all.append(compute_motor_sample_id(epoch_id))
                subj = extract_motor_subject_id(file_path)
                if subj is None:
                    raise ValueError(f"Cannot extract subject id from {file_path}")
                subject_ids_all.append(int(subj[3:]))

        total_num = len(trials_all)

        # 输出统计信息
        if skipped_30ch > 0:
            print(f"ℹ️  跳过 {skipped_30ch} 个30通道的样本")
        if skipped_other > 0:
            print(f"⚠️  跳过 {skipped_other} 个其他问题的样本（通道数/时间长度/标签缺失等）")

        if total_num == 0:
            raise ValueError("没有成功加载任何样本！请检查数据路径和格式。")

        if self.return_sample_id and len(set(sample_ids_all)) != len(sample_ids_all):
            raise ValueError(
                "Duplicate sample_id detected in Motor6ClassDataset; "
                "check for duplicate/conflicting epoch files."
            )

        # Stack所有trial: (Total_Trials, 22, 500)
        trials_all_arr = np.stack(trials_all, axis=0)
        if self.do_normalization:
            trials_all_arr = self.normalize(trials_all_arr)

        return (
            trials_all_arr,
            np.array(labels_all).flatten(),
            sample_ids_all,
            subject_ids_all,
            total_num,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        """
        trial_data_normalized = self.trials[idx]  # (22, 500)
        label = self.labels[idx]  # int (0-5)

        result = self.preprocess_sample(
            sample=trial_data_normalized,
            seq_len=self.num_chunks,  # 2
            labels=label,
        )
        if self.return_sample_id:
            result["sample_id"] = self.sample_ids[idx]
        return result


class Emotion7ClassDataset(EEGDataset):
    """
    Emotion 6分类数据集类（移除neutral类别）
    - 62通道输入，转换为22通道
    - 6分类任务 (0-5): happy=0, sad=1, disgust=2, fear=3, surprise=4, anger=5
    - 跳过neutral类别（原始label=2）
    - 标签映射：happy=0→0, sad=1→1, disgust=3→2, fear=4→3, surprise=5→4, anger=6→5
    - 4秒数据（1000个时间点@250Hz），分成4个chunks（每个chunk 250个时间点，1秒）
    """
    def __init__(
        self,
        filenames,
        sample_keys,
        chunk_len=250,  # 1秒数据，用于分成4个chunks
        num_chunks=4,  # 4秒数据分成4个chunks
        ovlp=0,  # 不重叠
        root_path="",
        matrix_p_path=None,
        gpt_only=True,
    ):
        super().__init__(
            filenames,
            sample_keys,
            chunk_len,
            num_chunks,
            ovlp,
            root_path=root_path,
            gpt_only=gpt_only,
        )

        # 加载通道转换矩阵 P (22, 62)
        try:
            self.P = np.load(matrix_p_path)
            print(f"Loaded P matrix for channel mapping with shape: {self.P.shape}")
            if self.P.shape[0] != 22:
                raise ValueError(
                    f"P matrix output channels must be 22, but got {self.P.shape[0]}."
                )
            if self.P.shape[1] != 62:
                raise ValueError(
                    f"P matrix input channels must be 62, but got {self.P.shape[1]}."
                )
        except FileNotFoundError:
            print(
                f"ERROR: P matrix not found at {matrix_p_path}. Cannot perform channel mapping."
                )
            raise

        self.trials, self.labels, self.epoch_ids, self.num_trials_per_file = self.get_trials_all()
        print(f"✓ 成功加载 {self.num_trials_per_file} 个样本")

    def __len__(self):
        return self.num_trials_per_file

    def map2pret(self, data: np.ndarray) -> np.ndarray:
        """
        将62通道数据映射到22通道
        data shape: (62, N_samples) → (22, N_samples)
        """
        if self.P is None:
            raise RuntimeError("P matrix is not loaded. Cannot map channels.")

        if self.P.shape[1] != data.shape[0]:
            raise ValueError(
                f"Channel mismatch! P matrix expects {self.P.shape[1]} input channels, "
                f"but the loaded data has {data.shape[0]} channels."
            )

        return np.matmul(self.P, data)  # Output shape: (22, N_samples)

    @staticmethod
    def extract_video_index(epoch_id: str) -> int:
        """从 epoch_id 中提取 video_index
        例如: 'subject_1_video_index_3_chunk001' -> 3
        """
        match = re.search(r'video_index_(\d+)_chunk', epoch_id)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def extract_subject_id(epoch_id: str) -> int:
        """从 epoch_id 中提取 subject_id
        例如: 'subject_1_video_index_3_chunk001' -> 1
        """
        match = re.search(r'subject_(\d+)_', epoch_id)
        if match:
            return int(match.group(1))
        return None

    def get_trials_all(self) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
        """
        加载所有数据并映射通道
        返回: (trials, labels, epoch_ids, num_trials)
        - 62通道数据映射到22通道
        - 4秒数据（1000个时间点）保持不变，用于分成4个chunks
        - 保存epoch_id信息用于投票评估
        """
        trials_all = []
        labels_all = []
        epoch_ids_all = []
        skipped_other = 0

        for filename in self.filenames:
            # self.filenames已经是完整路径（由base.py处理）
            file_path = filename
            try:
                with open(file_path, "rb") as f:
                    sample = pickle.load(f)
            except Exception as e:
                skipped_other += 1
                continue

            # 提取数据
            if "signal" in sample:
                trial_data = sample["signal"]  # shape: (channels, time)
            else:
                skipped_other += 1
                continue

            # 检查通道数：应该是62通道
            if trial_data.shape[0] != 62:
                skipped_other += 1
                continue

            # 检查时间长度：应该是1000个时间点（4秒@250Hz）
            if trial_data.shape[1] != 1000:
                skipped_other += 1
                continue

            if "label" in sample:
                label = sample["label"]
            else:
                skipped_other += 1
                continue

            # 跳过neutral类别（label=2）
            if label == 2:
                skipped_other += 1
                continue

            # 重新映射标签：移除neutral=2后的映射
            # 原始：happy=0, sad=1, neutral=2, disgust=3, fear=4, surprise=5, anger=6
            # 新：  happy=0, sad=1, disgust=2, fear=3, surprise=4, anger=5
            if label > 2:
                label = label - 1  # 3→2, 4→3, 5→4, 6→5

            # 提取epoch_id
            if "epoch_id" in sample:
                epoch_id = sample["epoch_id"]
            else:
                # 如果pickle中没有epoch_id，尝试从文件名提取
                filename_base = os.path.basename(file_path)
                epoch_id = filename_base.replace(".pickle", "").replace(".pkl", "")
            
            # 映射到22通道
            mapped_data = self.map2pret(trial_data)  # (22, 1000)

            trials_all.append(mapped_data)
            labels_all.append(label)
            epoch_ids_all.append(epoch_id)

        total_num = len(trials_all)
        
        # 输出统计信息
        skipped_neutral = 0
        for filename in self.filenames:
            try:
                with open(filename, "rb") as f:
                    sample = pickle.load(f)
                if "label" in sample and sample["label"] == 2:
                    skipped_neutral += 1
            except:
                pass
        
        if skipped_neutral > 0:
            print(f"ℹ️  跳过 {skipped_neutral} 个neutral类别（label=2）的样本")
        if skipped_other > 0:
            print(f"⚠️  跳过 {skipped_other} 个其他问题的样本（通道数/时间长度/标签缺失等）")

        if total_num == 0:
            raise ValueError("没有成功加载任何样本！请检查数据路径和格式。")

        # Stack所有trial: (Total_Trials, 22, 1000)
        trials_all_arr = np.stack(trials_all, axis=0)

        return self.normalize(trials_all_arr), np.array(labels_all).flatten(), epoch_ids_all, total_num

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        """
        trial_data_normalized = self.trials[idx]  # (22, 1000)
        label = self.labels[idx]  # int (0-5, 已移除neutral)
        epoch_id = self.epoch_ids[idx]  # str

        result = self.preprocess_sample(
            sample=trial_data_normalized,
            seq_len=self.num_chunks,
            labels=label,
        )
        
        # 添加epoch_id信息（用于投票评估）
        result['epoch_id'] = epoch_id
        
        return result


class Sleep5ClassDataset(EEGDataset):
    """
    Sleep 5分类数据集类
    - 6通道输入（C3, C4, F3, F4, O1, O2），转换为22通道
    - 5分类任务 (0, 1, 2, 3, 4)
    - 30秒数据（7500个时间点@250Hz），分成30个chunks（每个250个时间点，1秒）
    """
    def __init__(
        self,
        filenames,
        sample_keys,
        chunk_len=250,  # 1秒数据，用于分成30个chunks
        num_chunks=30,  # 30秒数据分成30个chunks
        ovlp=0,  # 不重叠
        root_path="",
        matrix_p_path=None,
        gpt_only=True,
    ):
        super().__init__(
            filenames,
            sample_keys,
            chunk_len,
            num_chunks,
            ovlp,
            root_path=root_path,
            gpt_only=gpt_only,
        )

        # 加载通道转换矩阵 P (22, 6)
        try:
            self.P = np.load(matrix_p_path)
            print(f"Loaded P matrix for channel mapping with shape: {self.P.shape}")
            if self.P.shape[0] != 22:
                raise ValueError(
                    f"P matrix output channels must be 22, but got {self.P.shape[0]}."
                )
            if self.P.shape[1] != 6:
                raise ValueError(
                    f"P matrix input channels must be 6, but got {self.P.shape[1]}."
                )
        except FileNotFoundError:
            print(
                f"ERROR: P matrix not found at {matrix_p_path}. Cannot perform channel mapping."
            )
            raise

        # 对于 sleep 数据集，数据量较大，不在初始化时一次性加载到内存
        # 只保存文件名列表，按需在 __getitem__ 中按索引读取单个样本，避免 OOM
        self.num_trials_per_file = len(self.filenames)
        print(f"✓ Sleep5ClassDataset 初始化完成，共 {self.num_trials_per_file} 个样本（按文件计）")

    def __len__(self):
        return self.num_trials_per_file

    def map2pret(self, data: np.ndarray) -> np.ndarray:
        """
        将6通道数据映射到22通道
        data shape: (6, N_samples) → (22, N_samples)
        """
        if self.P is None:
            raise RuntimeError("P matrix is not loaded. Cannot map channels.")

        if self.P.shape[1] != data.shape[0]:
            raise ValueError(
                f"Channel mismatch! P matrix expects {self.P.shape[1]} input channels, "
                f"but the loaded data has {data.shape[0]} channels."
            )

        return np.matmul(self.P, data)  # Output shape: (22, N_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        """
        filename = self.filenames[idx]
        file_path = os.path.join(self.root_path, filename)

        try:
            with open(file_path, "rb") as f:
                sample = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"加载样本失败: {file_path}, 错误信息: {e}")

        # 提取数据
        if "signal" not in sample:
            raise KeyError(f"样本 {file_path} 中缺少 'signal' 键")

        trial_data = sample["signal"]  # shape: (channels, time)

        # 基本检查
        if trial_data.shape[0] != 6:
            raise ValueError(f"样本 {file_path} 的通道数不是 6，而是 {trial_data.shape[0]}")

        if trial_data.shape[1] != 7500:
            raise ValueError(f"样本 {file_path} 的时间长度不是 7500，而是 {trial_data.shape[1]}")

        if "label" not in sample:
            raise KeyError(f"样本 {file_path} 中缺少 'label' 键")

        label = sample["label"]

        if label not in [0, 1, 2, 3, 4]:
            raise ValueError(f"样本 {file_path} 的标签 {label} 不在 [0,1,2,3,4] 范围内")

        # 映射到22通道 (22, 7500)
        mapped_data = self.map2pret(trial_data)

        # 归一化（复用 EEGDataset.normalize）
        trial_data_normalized = self.normalize(mapped_data[np.newaxis, ...])[0]

        return self.preprocess_sample(
            sample=trial_data_normalized,
            seq_len=self.num_chunks,  # 30
            labels=label,
        )
