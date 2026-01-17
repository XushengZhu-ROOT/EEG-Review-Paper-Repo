import pickle
import os
import numpy as np

file_path = '/home/dung/Documents/EEG-Review-Paper-Repo/isruc_labram/train/sub09_ep0406.pickle'
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(f"Keys: {list(data.keys())}")
        print(f"Label: {data['label']}")
        print(f"Label type: {type(data['label'])}")
except Exception as e:
    print(f"Error reading file: {e}")
