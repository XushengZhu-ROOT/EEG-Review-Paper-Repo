# %%
import mne
import numpy as np
import pandas as pd
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
#eg: kaggleERN
# csv_path = Path("ChannelsLocation.csv")
# df = pd.read_csv(csv_path)
# down_ch_names = df["Labels"].tolist()   

# motor6class (20ch)，顺序来自 datamake/Sub01_Clean/Clean/Sub01_8fast_clean.mat 的 EEG.chanlocs
down_ch_names = [
    "F7", "Fp1", "Fp2", "F8", "F3", "Fz", "F4", "C3", "Cz", "P8",
    "P7", "Pz", "P4", "T3", "P3", "O1", "O2", "C4", "T4", "A2",
]

sfreq = 250.0
montage = mne.channels.make_standard_montage("standard_1020")
info_pre = mne.create_info(tuh_ch_names, sfreq, ch_types="eeg")
info_pre.set_montage(montage)
info_down = mne.create_info(down_ch_names, sfreq, ch_types="eeg")
info_down.set_montage(montage)

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
G_down = fwd_down["sol"]["data"]  # [56, S]

print(G_pre.shape, G_down.shape)

# %%

lam = 0.05   

Gd = G_down          # [your channel number, S]
Gt = Gd.T            # [S, your channel number]

G_down_pinv = Gt @ np.linalg.inv(Gd @ Gt + lam * np.eye(Gd.shape[0]))  # [S, 56]

T = G_pre @ G_down_pinv   # [22, 56]

print("T shape:", T.shape)

# SAVE
np.save("../tMatrix_22x20_motor.npy", T)


# %%



