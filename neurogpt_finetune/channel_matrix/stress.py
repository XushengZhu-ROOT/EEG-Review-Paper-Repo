# %%
import mne
import numpy as np
from pathlib import Path

# neuroGPT target channel（do not change）
tuh_ch_names = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FT9", "T3", "C3", "Cz", "C4", "T4", "FT10",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "Oz", "O2",
]

# your channel in order
# stress (30ch)，顺序来自 stress_data/_run_labram_preprocess.py 的
# subset_channels 列表 -- 与原始 EDF 通道顺序一致（raw.pick_channels 对这份
# 列表不会重排，已用 stress_data/increase_edf_no400/1.edf 验证过）。
down_ch_names = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]

sfreq = 250.0
montage = mne.channels.make_standard_montage("standard_1020")
info_pre = mne.create_info(tuh_ch_names, sfreq, ch_types="eeg")
info_pre.set_montage(montage)
info_down = mne.create_info(down_ch_names, sfreq, ch_types="eeg")
# stress 通道名全大写（'FP1'/'FZ'...），standard_1020 montage 里是 'Fp1'/'Fz'
# 这种大小写混合形式，match_case=False 让匹配按大小写不敏感比对。
info_down.set_montage(montage, match_case=False)

import os
import mne

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)  #  .../fsaverage
subjects_dir = os.path.dirname(fs_dir)              #  .../MNE-fsaverage-data
subject = "fsaverage"

src = mne.setup_source_space(
    subject=subject,
    spacing="oct6",
    add_dist=False,
    subjects_dir=subjects_dir,
    verbose=True,
)

bem_model = mne.make_bem_model(
    subject=subject,
    ico=4,
    conductivity=(0.3, 0.006, 0.3),   #  brain, skull, scalp
    subjects_dir=subjects_dir,
)
bem = mne.make_bem_solution(bem_model)

trans = "fsaverage"

fwd_pre = mne.make_forward_solution(
    info=info_pre,
    trans=trans,
    src=src,
    bem=bem,
    meg=False,
    eeg=True,
    mindist=5.0,
    n_jobs=1,
)

fwd_down = mne.make_forward_solution(
    info=info_down,
    trans=trans,
    src=src,
    bem=bem,
    meg=False,
    eeg=True,
    mindist=5.0,
    n_jobs=1,
)

G_pre = fwd_pre["sol"]["data"]    #  [22, S]
G_down = fwd_down["sol"]["data"]  # [30, S]

print(G_pre.shape, G_down.shape)

# %%

lam = 0.05

Gd = G_down          # [your channel number, S]
Gt = Gd.T            # [S, your channel number]

G_down_pinv = Gt @ np.linalg.inv(Gd @ Gt + lam * np.eye(Gd.shape[0]))  # [S, 30]

T = G_pre @ G_down_pinv   # [22, 30]

print("T shape:", T.shape)

# SAVE
np.save("../tMatrix_22x30_stress.npy", T)
