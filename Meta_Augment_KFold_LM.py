import os
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import json

# === 路徑設定 ===
AUDIO_BASE = r"C:\Users\brad9\Desktop\BirdCLEF\train_audio"        # 改為你的音檔資料夾
TRAIN_CSV = r"C:\Users\brad9\Desktop\BirdCLEF\train.csv"     # 改為你的 train.csv
TAXO_CSV = r"C:\Users\brad9\Desktop\BirdCLEF\taxonomy.csv"   # 改為你的 taxonomy.csv
META_OUT = "meta_all.csv"
H5_OUT = "train_Augment_KFold_LM.h5"

# === 1. 讀取資料 & 合併 taxonomy ===
train_df = pd.read_csv(TRAIN_CSV)
taxonomy_df = pd.read_csv(TAXO_CSV)
df = train_df.merge(taxonomy_df[['primary_label', 'class_name']], on='primary_label', how='left')

# === 2. 計算 duration ===
def get_duration(row):
    audio_path = os.path.join(AUDIO_BASE, row['filename'])
    try:
        info = sf.info(audio_path)
        return info.duration
    except:
        return 0
tqdm.pandas(desc="音訊秒數計算")
df['duration'] = df.progress_apply(get_duration, axis=1)

# === 3. 計算每個物種總秒數 & 分三級 ===
species_duration = df.groupby('class_name')['duration'].sum().to_dict()
df['species_total_seconds'] = df['class_name'].map(species_duration)
# 三分位
species_sec = list(species_duration.values())
th_low = np.percentile(species_sec, 33)
th_high = np.percentile(species_sec, 66)
def get_aug_level(sec):
    if sec <= th_low: return 'A'
    elif sec <= th_high: return 'B'
    else: return 'C'
df['augment_level'] = df['species_total_seconds'].apply(get_aug_level)

# === 4. Onehot 標記 ===
all_labels = sorted(df['primary_label'].unique())
label2idx = {label: i for i, label in enumerate(all_labels)}
def encode_onehot(label):
    onehot = np.zeros(len(all_labels), dtype=int)
    onehot[label2idx[label]] = 1
    return ','.join(map(str, onehot))
df['onehot'] = df['primary_label'].apply(encode_onehot)

# === 5. 片段切分（每5秒一片段，segment_id，原有有切片則維持）===
SEG_DUR = 5
SAMPLE_RATE = 32000
meta_records = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    n_slices = int(np.ceil(row['duration'] / SEG_DUR))
    for seg_idx in range(n_slices):
        meta_records.append({
            'filename': row['filename'],
            'segment_id': f"{row['filename']}_seg{seg_idx}",
            'primary_label': row['primary_label'],
            'class_name': row['class_name'],
            'duration': row['duration'],
            'rating': row['rating'] if 'rating' in row else 5,
            'onehot': row['onehot'],
            'species_total_seconds': row['species_total_seconds'],
            'augment_level': row['augment_level']
        })
meta_df = pd.DataFrame(meta_records)

# === 6. GroupKFold（音檔為單位，每fold都保有所有物種）===
N_FOLDS = 3
gkf = GroupKFold(n_splits=N_FOLDS)
meta_df['fold'] = -1
file_groups = meta_df[['filename', 'class_name']].drop_duplicates()
for fold, (_, val_idx) in enumerate(gkf.split(file_groups, groups=file_groups['filename'])):
    valid_files = set(file_groups.iloc[val_idx]['filename'])
    is_valid = meta_df['filename'].isin(valid_files)
    meta_df.loc[is_valid, 'fold'] = fold

# === 7. train/valid split 標記（每fold）===
meta_df['split'] = "train"
for fold in range(N_FOLDS):
    mask = (meta_df['fold'] == fold)
    meta_df.loc[mask, 'split'] = "valid"

# === 8. （可選）補足valid中某物種不存在的情形（如有缺失可over-sample訓練集到valid，但建議print報告不補valid）
for fold in range(N_FOLDS):
    fold_df = meta_df[meta_df['fold']==fold]
    missing_species = set(meta_df['class_name']) - set(fold_df['class_name'])
    if missing_species:
        print(f"Fold{fold} 缺少物種：{missing_species}")
        # 若需要自動補valid可在這裡補充，預設僅訓練集補

# === 9. 儲存 meta ===
meta_df.to_csv(META_OUT, index=False)
print("Meta 完成！")
