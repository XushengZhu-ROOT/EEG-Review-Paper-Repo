import os
import pdb
import re
import shutil
from collections import defaultdict

import h5py
import numpy as np
import gzip
import pickle
import time
import pandas as pd

def load_tuh_all(path):
    # files = os.listdir(path)
    filepath = []
    file=""
    # for file in files:
    groups = os.listdir(path)
    for group in groups:
        if os.path.isdir(os.path.join(path, group)):
            subs = os.listdir(os.path.join(path, file, group))
        else:
            continue
        for sub in subs:
            sessions = os.listdir(os.path.join(path, file, group, sub))
            for sess in sessions:
                montages = os.listdir(os.path.join(path, file, group, sub, sess))
                for mont in montages:
                    edf_files = os.listdir(os.path.join(path, file, group, sub, sess, mont))
                    for edf in edf_files:
                        full_path = os.path.join(path, file, group, sub, sess, mont, edf)
                        filepath.append(full_path)
                        # pdb.set_trace()
                        shutil.move(full_path, os.path.join(path, group, sess + "_" + mont + "_" + edf))
                        # pdb.set_trace()
                # load_eeg(filepath[-1])
    return filepath


def load_pickle(filename):
    start_time = time.time()
    with gzip.open(filename, "rb") as file:
        data = pickle.load(file)
    print(data)
    end_time = time.time()
    print("Compressed Elapsed time:", end_time - start_time, "seconds")
    
    return data['data'], np.array(data['channel'])
  

def read_threshold_sub(csv_file, lower_bound=2599, upper_bound=1000000):
    df_read = pd.read_csv(csv_file)
    # Access the list of filenames and time_len
    filenames = df_read['filename'].tolist()
    time_lens = df_read['time_len'].tolist()
    filtered_files = []
    for fn, tlen in zip(filenames, time_lens):
        if (tlen > lower_bound) and (tlen < upper_bound):
            filtered_files.append(fn)
    return filtered_files

def get_epi_files(path, epi_csv, nonepi_csv, lower_bound=2599, upper_bound=1000000):
    epi_full_path = []
    nonepi_full_path = []
    if epi_csv is not None:
        epi_filtered_files = read_threshold_sub(epi_csv, lower_bound, upper_bound)
        epi_full_path = [path + "/epilepsy_edf/" + fn for fn in epi_filtered_files]
    if nonepi_csv is not None:
        nonepi_filtered_files = read_threshold_sub(nonepi_csv, lower_bound, upper_bound)
        nonepi_full_path = [path + "/no_epilepsy_edf/" + fn for fn in nonepi_filtered_files]

    return epi_full_path + nonepi_full_path

def read_sub_list(epi_list):
    with open(epi_list, 'r') as file:
        items = file.readlines()
    # Remove newline characters
    epi_subs = [item.strip() for item in items]
    return epi_subs

def exclude_epi_subs(csv_file, epi_list, lower_bound=2599, upper_bound=1000000, files_all=None):
    epi_subs = read_sub_list(epi_list)
    group_epi_subs = epi_subs
    if files_all is None:
        all_files = read_threshold_sub(csv_file, lower_bound, upper_bound)
    else:
        all_files = files_all
    filtered_files = [f for f in all_files if not any(sub_id in f for sub_id in group_epi_subs)]
    # pdb.set_trace()
    return filtered_files

def exclude_sz_subs(csv_file, lower_bound=2599, upper_bound=1000000, files_all=None):
    if files_all is None:
        all_files = read_threshold_sub(csv_file, lower_bound, upper_bound)
    else:
        all_files = files_all
    with open('sz_subs.txt', 'r') as f:
        sz_subs = f.readlines()
    filtered_files = [f for f in all_files if not any(sub_id in f for sub_id in sz_subs)]
    # pdb.set_trace()
    return filtered_files


# ===== [LOSO] Motor task sample_id -- ported verbatim (same regex/offsets) from
# labram_finetune/utils.py so that the sample_id set computed here for a given
# underlying AllSubjects_Epochs pickle file is byte-identical to what every other
# already-converted model (cbramod/biot/eegpt/neurolm/labram) computes for that
# same file. Do not change the constants/regex without updating all sibling
# model dirs in lockstep -- cross-model comparisons depend on this matching. =====
_MOTOR_SAMPLE_ID_TASK_ORDER = ['Walk', '8', 'Horizontal', 'Vertical', 'Pick', 'Stair']
_MOTOR_SAMPLE_ID_SPEED_ORDER = ['slow', 'medium', 'fast']
_MOTOR_SAMPLE_ID_TASK_OFFSET = 3000
_MOTOR_SAMPLE_ID_SPEED_OFFSET = 1000
_MOTOR_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(.+?)_epoch(\d+)$')
_MOTOR_SUBJECT_RE = re.compile(r'(Sub\d+)_')


def _parse_motor_task_token(task_token):
    for speed in _MOTOR_SAMPLE_ID_SPEED_ORDER:
        if task_token.endswith(speed):
            return task_token[: -len(speed)], speed
    raise ValueError(f"Cannot parse speed suffix (slow/medium/fast) from task token: {task_token!r}")


def compute_motor_sample_id(epoch_id):
    """由 epoch_id（如 'Sub04_Walkslow_epoch009'）确定性地生成 sample_id，
    格式：S{subject:02d}_ep{index:05d}（Motor 无 session 概念，不加 sess 段）。"""
    m = _MOTOR_SAMPLE_ID_RE.match(epoch_id)
    if not m:
        raise ValueError(f"Cannot parse epoch_id for sample_id: {epoch_id!r}")
    subject_num = int(m.group(1))
    task_token = m.group(2)
    local_idx = int(m.group(3))
    base_task, speed = _parse_motor_task_token(task_token)
    if base_task not in _MOTOR_SAMPLE_ID_TASK_ORDER:
        raise ValueError(
            f"Unknown base task {base_task!r} parsed from epoch_id {epoch_id!r}; "
            f"expected one of {_MOTOR_SAMPLE_ID_TASK_ORDER}"
        )
    task_idx = _MOTOR_SAMPLE_ID_TASK_ORDER.index(base_task)
    speed_idx = _MOTOR_SAMPLE_ID_SPEED_ORDER.index(speed)
    global_index = task_idx * _MOTOR_SAMPLE_ID_TASK_OFFSET + speed_idx * _MOTOR_SAMPLE_ID_SPEED_OFFSET + local_idx
    if global_index > 99999:
        raise ValueError(f"sample_id index overflow (>99999) for epoch_id {epoch_id!r}: {global_index}")
    return f"S{subject_num:02d}_ep{global_index:05d}"


def extract_motor_subject_id(name):
    """'Sub04_8fast_epoch001.pickle'（或包含它的任意路径）-> 'Sub04'。"""
    m = _MOTOR_SUBJECT_RE.search(os.path.basename(name))
    return m.group(1) if m else None


# ===== [LOSO] Stress task sample_id -- ported verbatim (same regex/format) from
# cbramod_finetune/datasets/custom_stress_dataset.py's compute_stress_sample_id
# so that the sample_id computed here for a given underlying chunk file is
# byte-identical to what every other already-converted model
# (cbramod/neurolm/labram/eegpt) computes for that same file. =====
_STRESS_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(increase|normal)_edf(\d+)_chunk(\d+)$')
_STRESS_SUBJECT_RE = re.compile(r'(Sub\d+)_')


def compute_stress_sample_id(chunk_id):
    """由 chunk_id（如 'Sub04_increase_edf27_chunk0012'）确定性地生成 sample_id，
    格式：S{subject:02d}_edf{edf_num}_chunk{local_idx:04d}。"""
    m = _STRESS_SAMPLE_ID_RE.match(chunk_id)
    if not m:
        raise ValueError(f"Cannot parse chunk_id for sample_id: {chunk_id!r}")
    subject_num = int(m.group(1))
    edf_num = int(m.group(3))
    local_idx = int(m.group(4))
    return f"S{subject_num:02d}_edf{edf_num}_chunk{local_idx:04d}"


def extract_stress_subject_id(name):
    """'Sub04_increase_edf27_chunk0012.pickle'（或包含它的任意路径）-> 'Sub04'。"""
    m = _STRESS_SUBJECT_RE.search(os.path.basename(name))
    return m.group(1) if m else None


# ===== [KaggleERN bestval] KaggleERN 的 epoch_id 命名跟 Stress 不一样（形如
# "S02_Sess01_FB004"，preprocess_KaggleERN_new.ipynb 存盘时用的就是这个当文件名），
# 不能套用 compute_stress_sample_id/extract_stress_subject_id 的 "SubNN_..." 正则，
# 这里单独写一份，跟 biot_finetune/finetune_evaluator.py 等已经在用的
# "^S(\\d+)_" 被试号解析约定保持一致。 =====
_KAGGLEERN_SAMPLE_ID_RE = re.compile(r'^S(\d+)_')
_KAGGLEERN_SUBJECT_RE = re.compile(r'^(S\d+)_')


def compute_kaggleern_sample_id(epoch_id):
    """KaggleERN 的 epoch_id 本来就是唯一、稳定、跨模型一致的标识（各架构的
    preprocess_KaggleERN_new.ipynb 都用它当文件名），不需要像 Motor/Stress 那样
    重新编号，原样返回即可；这里只做一次格式校验，防止传入意料之外的文件名。"""
    if not _KAGGLEERN_SAMPLE_ID_RE.match(epoch_id):
        raise ValueError(f"Cannot parse epoch_id for sample_id: {epoch_id!r}")
    return epoch_id


def extract_kaggleern_subject_id(name):
    """'S02_Sess01_FB004.pickle'（或包含它的任意路径）-> 'S02'。"""
    m = _KAGGLEERN_SUBJECT_RE.search(os.path.basename(name))
    return m.group(1) if m else None


def list_stress_files_by_subject(root):
    """[LOSO] 扫描 root/{train,val,test}/*.pickle，按受试者分组（'Sub04' -> [abs_path, ...]），
    用于构造 Stress 任务的受试者独立（LOSO）划分。返回绝对路径。"""
    subject_to_files = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".pickle"):
                continue
            sid = extract_stress_subject_id(fname)
            if sid is None:
                continue
            subject_to_files[sid].append(os.path.abspath(os.path.join(split_dir, fname)))
    return subject_to_files


def list_motor_files_by_subject(root):
    """[LOSO] 扫描 root/{train,val,test}/*.pickle，按受试者分组（'Sub04' -> [abs_path, ...]），
    用于构造受试者独立（LOSO）划分。返回绝对路径，方便直接喂给 EEGDataset
    （EEGDataset 在 root_path 非空时会对 filenames 再 os.path.join 一次，
    绝对路径能保证这次多余的 join 是无害的 no-op）。"""
    subject_to_files = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".pickle"):
                continue
            sid = extract_motor_subject_id(fname)
            if sid is None:
                continue
            subject_to_files[sid].append(os.path.abspath(os.path.join(split_dir, fname)))
    return subject_to_files

