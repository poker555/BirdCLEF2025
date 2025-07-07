import pandas as pd

# 讀取csv
df = pd.read_csv(r'C:\Users\brad9\Desktop\BirdCLEF\log_mel.csv')

# 假設mel_filename格式為 1139490\CSA36385_seg0.npy
# Windows分隔符號是 \
df['folder_id'] = df['mel_filename'].apply(lambda x: str(x).split('\\')[0])

# 檢查結果
print(df[['mel_filename', 'folder_id']].head())

# 儲存新檔案
df.to_csv(r'C:\Users\brad9\Desktop\BirdCLEF\log_mel_v2.csv', index=False)