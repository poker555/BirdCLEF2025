import h5py
import pandas as pd
import numpy as np

# ==== 請根據你的實際檔案路徑調整 ====
h5_path = r"C:\Users\brad9\Desktop\BirdCLEF\train_valid_weighted.h5"
taxonomy_path = r"C:\Users\brad9\Desktop\BirdCLEF\taxonomy.csv"

# ==== 讀 taxonomy 並建立 label_to_idx (必須和你建立 onehot 的排序一致) ====
taxonomy = pd.read_csv(taxonomy_path)
all_labels = sorted(taxonomy['primary_label'].astype(str).unique())
label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

# ==== 讀 h5 資料 ====
with h5py.File(h5_path, 'r') as h5f:
    print("檔案內容:")
    for key in h5f.keys():
        print(f"{key}: {h5f[key].shape}")
