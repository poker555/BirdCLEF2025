import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# === 參數區 ===
MEL_DIR = "C:/Users/brad9/Desktop/BirdCLEF/log_mel_32000kHZ"
CSV_PATH = "C:/Users/brad9/Desktop/BirdCLEF/updated_train_mel.csv"   # 你的CSV檔
HDF5_PATH = "C:/Users/brad9/Desktop/BirdCLEF/all_mel_data.h5"
RECORD_PATH = "C:/Users/brad9/Desktop/BirdCLEF/hdf5_index_record.csv"
ORG_IDX_PATH = "C:/Users/brad9/Desktop/BirdCLEF/index/organism_class_to_idx.json"

# 讀取對應的class映射
with open(ORG_IDX_PATH) as f:
    organism_class_to_idx = json.load(f)
num_classes = len(organism_class_to_idx)

# 讀CSV
df = pd.read_csv(CSV_PATH)
mel_files = df['mel_filename'].tolist()

# 支援 multi-label（如沒有則 fallback 為 primary_label）
def encode_labels(row):
    label_vec = np.zeros(num_classes, dtype=np.float32)
    if "multi_labels" in row and not pd.isnull(row["multi_labels"]):
        for name in str(row["multi_labels"]).split(','):
            name = name.strip()
            if name in organism_class_to_idx:
                label_vec[organism_class_to_idx[name]] = 1
    else:
        label = row["primary_label"]
        if label in organism_class_to_idx:
            label_vec[organism_class_to_idx[label]] = 1
    return label_vec

labels = np.stack([encode_labels(row) for _, row in df.iterrows()])

# 取出 shape
mel_sample = np.load(os.path.join(MEL_DIR, mel_files[0].replace("\\", "/")))
if mel_sample.shape[0] != 128:
    mel_sample = mel_sample.T
C, F, T = 1, mel_sample.shape[0], mel_sample.shape[1]
N = len(mel_files)

# === 儲存 HDF5 ===
with h5py.File(HDF5_PATH, "w") as h5f:
    mel_ds = h5f.create_dataset("mel", shape=(N, C, F, T), dtype=np.float32)
    label_ds = h5f.create_dataset("label", shape=(N, num_classes), dtype=np.float32)
    record = []
    for i, fname in enumerate(tqdm(mel_files)):
        mel_path = os.path.join(MEL_DIR, fname.replace("\\", "/"))
        mel = np.load(mel_path)
        if mel.shape[0] != 128:
            mel = mel.T
        mel = mel[np.newaxis, :, :]
        mel_ds[i] = mel.astype(np.float32)
        label_ds[i] = labels[i]
        record.append({'index': i, 'mel_filename': fname})
pd.DataFrame(record).to_csv(RECORD_PATH, index=False)

print(f"轉換完成，共儲存 {N} 筆 mel。HDF5: {HDF5_PATH}，記錄: {RECORD_PATH}")
