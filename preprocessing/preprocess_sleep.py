# %%
import os
import mne
# mne.set_log_level("WARNING")
mne.set_log_level("ERROR")
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.model_selection import train_test_split
from scipy.signal import detrend


# %%
import sys
print(sys.executable)
print(mne.__version__)
print(mne.__file__)

# %%
# Paths for input csv folders
train_data_path = 'train'
chunk_size = 3.0
LINE_NOISE = 50.0

rs = 42 # seed number for data split 
nchan = 6 # number of channels

# %%
import os
import numpy as np
import mne

ISRUC_RAW_PATH = "./data/ISRUC_S3/RawData"
ISRUC_CHANNELS = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1']

def load_isruc_raw(sub_id, path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS):
    """
    读单个 ISRUC 被试的 EDF, 返回 MNE Raw, 只保留指定通道
    """
    ch_names = [c.replace("_", "-") for c in channels]
    edf_path = os.path.join(path_raw, f"{sub_id}/{sub_id}.edf")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.pick_channels(ch_names)
    return raw

def load_isruc_labels(sub_id, path_raw=ISRUC_RAW_PATH):

    labels = []
    label_path = os.path.join(path_raw, f"{sub_id}/{sub_id}_1.txt")
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            labels.append(int(line))
    labels = np.array(labels)
    labels[labels == 5] = 4   
    return labels

def slice_isruc_epochs(raw, labels, epoch_len_sec=30):

    sfreq = raw.info["sfreq"]
    epoch_len_samples = int(epoch_len_sec * sfreq)

    # data = raw.get_data()  # (n_channels, n_samples)
    data = raw.get_data() * 1e6   #  microvolt
    n_by_signal = data.shape[1] // epoch_len_samples
    n_by_label = len(labels)
    n_epochs = min(n_by_signal, n_by_label)

    if n_epochs == 0:
        print("Warning: no full epochs for this subject")
        return np.empty((0, raw.info["nchan"], epoch_len_samples)), np.array([])

    data = data[:, : n_epochs * epoch_len_samples]
    data_epochs = data.reshape(raw.info["nchan"], n_epochs, epoch_len_samples)
    data_epochs = np.transpose(data_epochs, (1, 0, 2))  # (n_epochs, n_channels, n_times)
    labels_used = labels[:n_epochs]

    return data_epochs, labels_used

# %%
# Function to save chunks as pickle files
def save_epochs(epochs, folder):
    for i, (epoch_data, label, epoch_id) in enumerate(epochs):
        
        sample = {
            'signal': epoch_data,  # shape: (n_channels, 800)
            'label': label,
            'epoch_id': epoch_id
        }
        
        filename = os.path.join(folder, f"{epoch_id}.pickle")
        
        with open(filename, 'wb') as f:
            pickle.dump(sample, f)
        print(f"Saved: {filename}")


# %%

def df_to_raw_full(df, sfreq=200):
    ch_names = [c for c in df.columns if c not in ['Time', 'FeedBackEvent', 'EOG']]
    eeg_data = df[ch_names].T.values  # (n_ch, n_times)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    return raw, ch_names


def filter_full(raw, l_freq, h_freq, line_noise=50, target_sfreq=200):
    sf = raw.info['sfreq']
    raw.filter(l_freq, h_freq, n_jobs=-1)
    raw.notch_filter(line_noise, filter_length='auto', n_jobs=-1)
    if target_sfreq and target_sfreq != sf:
        raw.resample(target_sfreq, npad='auto', verbose=False)
    return raw

# %%
def remove_dc_offset(raw):
    """Remove DC offset per channel."""
    data, times = raw.get_data(return_times=True)
    data -= np.mean(data, axis=1, keepdims=True)
    raw._data = data
    return raw

def remove_linear_trend(raw):
    """Remove linear trend per channel."""
    try:
        data = detrend(raw.get_data(), axis=1, type='linear')
        # print("✅ detrend finished successfully")
    except Exception as e:
        print("❌ detrend failed:", e)
    raw._data = data
    return raw


def normalize_amplitude(raw, mode="div100"):
    """Amplitude normalization according to mode."""
    data = raw.get_data()
    if mode == "div100":
        raw._data = data / 100.0  # µV
    elif mode == "zscore_per_channel":
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        raw._data = (data - mean) / std
    elif mode == "global_zscore":
        raw._data = (data - np.mean(data)) / np.std(data)
    elif mode == "percentile_95":
        p95 = np.percentile(np.abs(data), 95, axis=1, keepdims=True)
        raw._data = data / p95
    return raw

# %%
# CBraMod 在 ISRUC 上的前处理和切分
def extract_isruc_epochs_cbramod(sub_id, path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS,
                                 epoch_len_sec=30, line_noise=50):
    raw = load_isruc_raw(sub_id, path_raw, channels)
    labels = load_isruc_labels(sub_id, path_raw)

    raw.resample(200)
    raw = filter_full(raw, 0.3, 75.0, line_noise=line_noise, target_sfreq=200)
    # raw = normalize_amplitude(raw, "div100")

    data_epochs, labels_epochs = slice_isruc_epochs(raw, labels, epoch_len_sec)

    epochs = []
    for ep_idx in range(len(labels_epochs)):
        signal = data_epochs[ep_idx]              # (n_channels, n_times)
        label = int(labels_epochs[ep_idx])        # 0..4
        epoch_id = f"sub{sub_id:02d}_ep{ep_idx:04d}"
        epochs.append((signal, label, epoch_id))

    print(f"[ISRUC CBraMod] sub{sub_id} -> {len(epochs)} epochs")
    if len(epochs) < 60:
        wired_file.append(f"ISRUC_sub{sub_id}_cbramod")
    return epochs


def extract_isruc_epochs_labram(sub_id, path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS,
                                epoch_len_sec=30, line_noise=50):
    raw = load_isruc_raw(sub_id, path_raw, channels)
    labels = load_isruc_labels(sub_id, path_raw)

    raw.resample(200)
    raw.filter(l_freq=0.1, h_freq=75.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)

    data_epochs, labels_epochs = slice_isruc_epochs(raw, labels, epoch_len_sec)

    epochs = []
    for ep_idx in range(len(labels_epochs)):
        signal = data_epochs[ep_idx]
        label = int(labels_epochs[ep_idx])
        epoch_id = f"sub{sub_id:02d}_ep{ep_idx:04d}"
        epochs.append((signal, label, epoch_id))

    print(f"[ISRUC LaBraM] sub{sub_id} -> {len(epochs)} epochs")
    if len(epochs) < 60:
        wired_file.append(f"ISRUC_sub{sub_id}_labram")
    return epochs


def extract_isruc_epochs_neurolm(sub_id, path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS,
                                 epoch_len_sec=30, line_noise=50):
    raw = load_isruc_raw(sub_id, path_raw, channels)
    labels = load_isruc_labels(sub_id, path_raw)

    raw.resample(200, npad="auto", verbose="error")
    raw.filter(l_freq=0.1, h_freq=75.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)

    data_epochs, labels_epochs = slice_isruc_epochs(raw, labels, epoch_len_sec)

    epochs = []
    for ep_idx in range(len(labels_epochs)):
        signal = data_epochs[ep_idx]
        label = int(labels_epochs[ep_idx])
        epoch_id = f"sub{sub_id:02d}_ep{ep_idx:04d}"
        epochs.append((signal, label, epoch_id))

    print(f"[ISRUC NeuroLM] sub{sub_id} -> {len(epochs)} epochs")
    if len(epochs) < 60:
        wired_file.append(f"ISRUC_sub{sub_id}_neurolm")
    return epochs


# BIOT 在 ISRUC 上
def extract_isruc_epochs_biot(sub_id, path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS,
                              epoch_len_sec=30, line_noise=50):
    raw = load_isruc_raw(sub_id, path_raw, channels)
    labels = load_isruc_labels(sub_id, path_raw)

    raw.resample(200, npad="auto", verbose="error")
    # raw.filter(l_freq=0.5, h_freq=45.0, n_jobs=1)
    # raw.notch_filter(line_noise, n_jobs=1)

    data_epochs, labels_epochs = slice_isruc_epochs(raw, labels, epoch_len_sec)

    epochs = []
    for ep_idx in range(len(labels_epochs)):
        signal = data_epochs[ep_idx]
        label = int(labels_epochs[ep_idx])
        epoch_id = f"sub{sub_id:02d}_ep{ep_idx:04d}"
        epochs.append((signal, label, epoch_id))

    print(f"[ISRUC BIOT] sub{sub_id} -> {len(epochs)} epochs")
    if len(epochs) < 60:
        wired_file.append(f"ISRUC_sub{sub_id}_biot")
    return epochs


def extract_isruc_epochs_eegpt(sub_id, path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS,
                               epoch_len_sec=30, line_noise=50):
    raw = load_isruc_raw(sub_id, path_raw, channels)
    labels = load_isruc_labels(sub_id, path_raw)

    raw.resample(256, npad="auto", verbose="error")
    raw.filter(l_freq=0.1, h_freq=75.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)
    data = raw.get_data()
    raw = remove_dc_offset(raw)
    #global average reference
    raw.set_eeg_reference(ref_channels='average', projection=False)

    data_epochs, labels_epochs = slice_isruc_epochs(raw, labels, epoch_len_sec)

    epochs = []
    for ep_idx in range(len(labels_epochs)):
        signal = data_epochs[ep_idx]
        label = int(labels_epochs[ep_idx])
        epoch_id = f"sub{sub_id:02d}_ep{ep_idx:04d}"
        epochs.append((signal, label, epoch_id))

    print(f"[ISRUC EEGPT] sub{sub_id} -> {len(epochs)} epochs")
    if len(epochs) < 60:
        wired_file.append(f"ISRUC_sub{sub_id}_eegpt")
    return epochs


def extract_isruc_epochs_neurogpt(sub_id, path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS,
                                  epoch_len_sec=30, line_noise=50):
    raw = load_isruc_raw(sub_id, path_raw, channels)
    labels = load_isruc_labels(sub_id, path_raw)

    raw.resample(250, npad="auto", verbose="error")
    raw.filter(l_freq=0.5, h_freq=100.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)
    raw = remove_dc_offset(raw)
    raw = remove_linear_trend(raw)
    raw.set_eeg_reference(ref_channels="average")
    # raw = normalize_amplitude(raw, "zscore_per_channel")

    data_epochs, labels_epochs = slice_isruc_epochs(raw, labels, epoch_len_sec)

    epochs = []
    for ep_idx in range(len(labels_epochs)):
        signal = data_epochs[ep_idx]
        label = int(labels_epochs[ep_idx])
        epoch_id = f"sub{sub_id:02d}_ep{ep_idx:04d}"
        epochs.append((signal, label, epoch_id))

    print(f"[ISRUC NeuroGPT] sub{sub_id} -> {len(epochs)} epochs")
    if len(epochs) < 60:
        wired_file.append(f"ISRUC_sub{sub_id}_neurogpt")
    return epochs


def extract_isruc_epochs_sttransformer(sub_id, path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS,
                                       epoch_len_sec=30, line_noise=50):
    raw = load_isruc_raw(sub_id, path_raw, channels)
    labels = load_isruc_labels(sub_id, path_raw)

    raw.resample(250, npad="auto", verbose="error")
    raw.filter(l_freq=4.0, h_freq=40.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)

    data_epochs, labels_epochs = slice_isruc_epochs(raw, labels, epoch_len_sec)

    epochs = []
    for ep_idx in range(len(labels_epochs)):
        signal = data_epochs[ep_idx]
        label = int(labels_epochs[ep_idx])
        epoch_id = f"sub{sub_id:02d}_ep{ep_idx:04d}"
        epochs.append((signal, label, epoch_id))

    print(f"[ISRUC ST-Transformer] sub{sub_id} -> {len(epochs)} epochs")
    if len(epochs) < 60:
        wired_file.append(f"ISRUC_sub{sub_id}_sttransformer")
    return epochs

# %%
ISRUC_EXTRACTORS = {
    "cbramod":        extract_isruc_epochs_cbramod,
    "labram":         extract_isruc_epochs_labram,
    "neurolm":        extract_isruc_epochs_neurolm,
    "biot":           extract_isruc_epochs_biot,
    "eegpt":          extract_isruc_epochs_eegpt,
    "neurogpt":       extract_isruc_epochs_neurogpt,
    "sttransformer":  extract_isruc_epochs_sttransformer,
}

def process_isruc_s3(model_name="neurogpt", epoch_len_sec=30, rs=42,
                     path_raw=ISRUC_RAW_PATH, channels=ISRUC_CHANNELS):

    extractor = ISRUC_EXTRACTORS[model_name]

    all_epochs = []
    for sub_id in range(1, 11):
        epochs_sub = extractor(sub_id, path_raw, channels, epoch_len_sec=epoch_len_sec)
        all_epochs.extend(epochs_sub)

    if len(all_epochs) == 0:
        print("No epochs extracted from ISRUC")
        return

    labels_all = [label for (_, label, _) in all_epochs]
    train_epochs, temp_epochs = train_test_split(
        all_epochs, test_size=0.2, random_state=rs, stratify=labels_all
    )
    labels_temp = [label for (_, label, _) in temp_epochs]
    val_epochs, test_epochs = train_test_split(
        temp_epochs, test_size=0.5, random_state=rs, stratify=labels_temp
    )

    random.seed(rs)
    random.shuffle(train_epochs)
    random.shuffle(val_epochs)
    random.shuffle(test_epochs)

    nchan_isruc = len(channels)
    out_root = f"isruc_{model_name}"
    train_folder = os.path.join(out_root, "train")
    val_folder   = os.path.join(out_root, "val")
    test_folder  = os.path.join(out_root, "test")
    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    save_epochs(train_epochs, train_folder)
    save_epochs(val_epochs,   val_folder)
    save_epochs(test_epochs,  test_folder)

    print(f"ISRUC processed with model {model_name}, saved to {out_root}")

# %%
# model_name="labram"  # models:"cbramod", "labram", "biot", "eegpt", "neurolm", "neurogpt", "sttransformer"
# process_isruc_s3(model_name)

# %%
import argparse

def main():
    parser = argparse.ArgumentParser(description="Preprocess ISRUC S3 for a specific model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(ISRUC_EXTRACTORS.keys()),
        help="Model name, for example labram or neurogpt.",
    )
    parser.add_argument(
        "--epoch_len",
        type=int,
        default=30,
        help="Epoch length in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split.",
    )
    args = parser.parse_args()

    process_isruc_s3(
        model_name=args.model,
        epoch_len_sec=args.epoch_len,
        rs=args.seed,
        path_raw=ISRUC_RAW_PATH,
        channels=ISRUC_CHANNELS,
    )

if __name__ == "__main__":
    main()

# %%
# %%bash
# python3 preprocess_sleep.py --model biot


