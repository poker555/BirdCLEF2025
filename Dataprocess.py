import pandas as pd
import torch
import os
import librosa
import soundfile as sf

train_audio = 'C:/Users/brad9/Desktop/BirdCLEF/train_audio/'
test_soundscape = 'C:/Users/brad9/Desktop/BirdCLEF/test_soundscape/'
train_soundscape = 'C:/Users/brad9/Desktop/BirdCLEF/train_soundscape/'

df = pd.read_csv('C:/Users/brad9/Desktop/BirdCLEF/train.csv')
print(df.head())

CSA_df = df[df['collection'] == 'CSA']

print(CSA_df.head())

for idx, row in CSA_df.iterrows():
    file_tail = row['filename']  
    full_path = os.path.join(train_audio, file_tail)
    
    if not os.path.exists(full_path):
        print(f"找不到檔案: {full_path}")
        continue

    try:
        # 讀取音訊
        y, sr = librosa.load(full_path, sr=None)
        
        # 前 8 秒
        y_short = y[:sr * 8]

        # 儲存路徑
        output_path = os.path.join(train_audio, os.path.basename(file_tail))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 儲存音訊
        sf.write(output_path, y_short, sr)
        print(f"已裁切: {file_tail}")

    except Exception as e:
        print(f"處理失敗 {file_tail}: {e}")

    

