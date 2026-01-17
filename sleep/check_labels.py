import os
import pickle
import numpy as np

root = '/home/dung/Documents/EEG-Review-Paper-Repo/isruc_labram/train'
files = os.listdir(root)
files.sort()

max_label = -1
min_label = 1000
labels = []

for i, file in enumerate(files):
    path = os.path.join(root, file)
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            label = data['label']
            labels.append(label)
            if label > max_label:
                max_label = label
            if label < min_label:
                min_label = label
    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"Checked {len(labels)} files.")
print(f"Max label: {max_label}")
print(f"Min label: {min_label}")
print(f"Unique labels: {np.unique(labels)}")
