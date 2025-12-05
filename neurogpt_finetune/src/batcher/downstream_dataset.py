import os
import pdb
import torch
import pickle
import numpy as np
from typing import Tuple, List, Dict
from batcher.base import EEGDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt


class MotorImageryDataset(EEGDataset):
    def __init__(
        self,
        filenames,
        sample_keys,
        chunk_len=500,
        num_chunks=10,
        ovlp=50,
        root_path="",
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

        self.data_all = []
        for fn in self.filenames:
            self.data_all.append(np.load(fn))

        self.mi_types = {
            769: "left",
            770: "right",
            771: "foot",
            772: "tongue",
            1023: "rejected",
        }  # , 783: 'unknown', 1023: 'rejected'
        # Types of motor imagery
        self.labels_string2int = {
            "left": 0,
            "right": 1,
            "foot": 2,
            "tongue": 3,
        }  # , 'unknown': -1
        self.Fs = 250  # 250Hz from original paper
        self.P = np.load("../inputs/tMatrix_value.npy")

        self.trials, self.labels, self.num_trials_per_sub = self.get_trials_all()
        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        return self.preprocess_sample(
            self.trials[idx], self.num_chunks, self.labels[idx]
        )

    def map2pret(self, data):
        return np.matmul(self.P, data)  # 22x22, 22xTime

    def get_trials_from_single_subj(self, sub_id):
        raw = self.data_all[sub_id]["s"].T
        events_type = self.data_all[sub_id]["etyp"].T
        events_position = self.data_all[sub_id]["epos"].T
        events_duration = self.data_all[sub_id]["edur"].T
        artifacts = self.data_all[sub_id]["artifacts"].T
        # Channel default is C3
        startrial_code = 768
        starttrial_events = events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trial_labels = self.get_labels(sub_id)

        trials = []
        classes = []
        for j, index in enumerate(idxs):
            try:
                # print(index)
                # type_e = events_type[0, index+1]
                # class_e = self.mi_types[type_e]
                # if type_e == 1023:
                #     continue
                # classes.append(self.labels_string2int[class_e])
                classes.append(trial_labels[j])

                start = events_position[0, index]
                stop = start + events_duration[0, index]
                trial = raw[:22, start + 500 : stop - 375]
                # add band-pass filter
                # self.bandpass_filter(trial, lowcut=4, highcut=40, fs=250, order=5)
                trials.append(trial)
            except:
                # print("Cannot load trial")
                continue
        return trials, classes

    def get_labels(self, sub_id):
        label_path = self.root_path + "true_labels/"
        base_name = os.path.basename(self.filenames[sub_id])
        sub_name = os.path.splitext(base_name)[0]
        labels = loadmat(label_path + sub_name + ".mat")["classlabel"]
        return labels.squeeze() - 1

    def get_trials_all(self):
        trials_all = []
        labels_all = []
        total_num = []
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            total_num.append(len(trials))
            trials_all.append(np.array(trials))
            labels_all.append(np.array(labels))
        # reordered_data = self.reorder_channels(np.vstack(trials_all))
        trials_all_arr = np.vstack(trials_all)
        # map to same channel configuration as pretraining
        trials_all_arr = self.map2pret(trials_all_arr)
        return self.normalize(trials_all_arr), np.array(labels_all).flatten(), total_num

    # def normalize(self, data):
    #     return (data - np.mean(data)) / np.std(data)

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a bandpass filter to the data.

        Parameters:
        - data: The EEG signal
        - lowcut: Low cut-off frequency
        - highcut: High cut-off frequency
        - fs: Sampling rate (frequency)
        - order: Order of the filter

        Returns:
        - Filtered data
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype="band")
        filtered_data = filtfilt(b, a, data)

        return filtered_data


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

        self.trials, self.labels, self.num_trials_per_file = self.get_trials_all()

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

    def get_trials_all(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        trials_all = []
        labels_all = []
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

        total_num = len(trials_all)

        # (Total_Trials, channel_size, chunk_len*sample_rate)
        trials_all_arr = np.stack(trials_all, axis=0)

        return self.normalize(trials_all_arr), np.array(labels_all).flatten(), total_num

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads pickle file, maps channels, and processes the sample.
        """
        trial_data_normalized_once = self.trials[idx]
        label = self.labels[idx]

        return self.preprocess_sample(
            sample=trial_data_normalized_once,
            seq_len=self.num_chunks,  # self.num_chunks is the 'seq_len' argument in preprocess_sample
            labels=label,  # Pass label as a numpy array for consistent handling
        )
