import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# === 參數區 ===
MEL_DIR = "C:/Users/brad9/Desktop/BirdCLEF/log_mel_32000kHZ"
CSV_PATH = "C:/Users/brad9/Desktop/BirdCLEF/code/merge_train.csv"  
HDF5_PATH = "C:/Users/brad9/Desktop/BirdCLEF/log_mel_data_v2.h5"
RECORD_PATH = "C:/Users/brad9/Desktop/BirdCLEF/log_mel_v2.csv"

# 讀CSV
df = pd.read_csv(CSV_PATH)
missing = df[df['mel_filename'].isnull() | (df['mel_filename'].astype(str).str.strip() == '')]
print("缺檔案名的資料：")
print(missing)
# 補齊缺值
df['mel_filename'] = df['mel_filename'].fillna('')

# 檢查型態
if not all(df['mel_filename'].apply(lambda x: isinstance(x, str))):
    print("有非字串內容！")
    print(df[~df['mel_filename'].apply(lambda x: isinstance(x, str))])
    
mel_files = df['mel_filename'].tolist()

# 四個類別標籤，補空字串
for col in ['class_name_Insecta', 'class_name_Amphibia', 'class_name_Mammalia', 'class_name_Aves']:
    df[col] = df[col].fillna('')

# 頻譜圖 shape
mel_sample = np.load(os.path.join(MEL_DIR, mel_files[0].replace("\\", "/")))
if mel_sample.shape[0] != 128:
    mel_sample = mel_sample.T
C, F, T = 1, mel_sample.shape[0], mel_sample.shape[1]
N = len(mel_files)

# --- 開始寫入 ---
with h5py.File(HDF5_PATH, "w") as h5f:
    mel_ds = h5f.create_dataset("mel", shape=(N, C, F, T), dtype=np.float32)
    class_label_ds = h5f.create_dataset("class_label", shape=(N,), dtype=np.int32)
    class_name_Insecta_ds = h5f.create_dataset("class_name_Insecta", shape=(N,), dtype='S32')
    class_name_Amphibia_ds = h5f.create_dataset("class_name_Amphibia", shape=(N,), dtype='S32')
    class_name_Mammalia_ds = h5f.create_dataset("class_name_Mammalia", shape=(N,), dtype='S32')
    class_name_Aves_ds = h5f.create_dataset("class_name_Aves", shape=(N,), dtype='S32')
    
    record = []
    for i, fname in enumerate(tqdm(mel_files)):
        mel_path = os.path.join(MEL_DIR, fname.replace("\\", "/"))
        mel = np.load(mel_path)
        if mel.shape[0] != 128:
            mel = mel.T
        mel = mel[np.newaxis, :, :]
        mel_ds[i] = mel.astype(np.float32)
        # 寫入標籤
        class_label_ds[i] = int(df.iloc[i]['class_label']) if pd.notnull(df.iloc[i]['class_label']) else -1
        class_name_Insecta_ds[i] = str(df.iloc[i]['class_name_Insecta']).encode('utf-8')
        class_name_Amphibia_ds[i] = str(df.iloc[i]['class_name_Amphibia']).encode('utf-8')
        class_name_Mammalia_ds[i] = str(df.iloc[i]['class_name_Mammalia']).encode('utf-8')
        class_name_Aves_ds[i] = str(df.iloc[i]['class_name_Aves']).encode('utf-8')
        record.append({
        'index': i,
        'mel_filename': fname,
        'class_label': df.iloc[i]['class_label'],
        'class_name_Insecta': df.iloc[i]['class_name_Insecta'],
        'class_name_Amphibia': df.iloc[i]['class_name_Amphibia'],
        'class_name_Mammalia': df.iloc[i]['class_name_Mammalia'],
        'class_name_Aves': df.iloc[i]['class_name_Aves'],

        })

pd.DataFrame(record).to_csv(RECORD_PATH, index=False)

print(f"轉換完成，共儲存 {N} 筆 mel。HDF5: {HDF5_PATH}，記錄: {RECORD_PATH}")
