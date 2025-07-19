import pandas as pd
import os
import librosa
from tqdm import tqdm

# 檔案路徑請根據實際情況調整
train_csv_path = r'C:\Users\brad9\Desktop\BirdCLEF\train.csv'
taxonomy_csv_path = r'C:\Users\brad9\Desktop\BirdCLEF\taxonomy.csv'
audio_root = r'C:\Users\brad9\Desktop\BirdCLEF\train_audio'  # 根據你的音檔目錄位置
output_path = r'C:\Users\brad9\Desktop\BirdCLEF/taxonomy_with_seconds.csv'

# 讀取資料
train_df = pd.read_csv(train_csv_path)
taxonomy_df = pd.read_csv(taxonomy_csv_path)

# 記錄每個 primary_label 的秒數
label_seconds = {}

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    audio_path = os.path.join(audio_root, row['filename'])
    try:
        duration = librosa.get_duration(filename=audio_path)
    except Exception as e:
        duration = 0
    label = row['primary_label']
    if label not in label_seconds:
        label_seconds[label] = 0
    label_seconds[label] += duration

# 新增一個欄位並根據 primary_label 填入秒數
taxonomy_df['total_seconds'] = taxonomy_df['primary_label'].map(label_seconds).fillna(0).astype(int)

# 儲存新的 taxonomy 檔案
taxonomy_df.to_csv(output_path, index=False)

print('已完成！結果存到:', output_path)
