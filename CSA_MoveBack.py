import pandas as pd
import shutil
import os

# === 使用者設定 ===
csv_path = r'C:\Users\brad9\Desktop\BirdCLEF\train.csv'                     # train.csv 路徑
processed_folder = r'C:\Users\brad9\Desktop\BirdCLEF\CSA_fix'       # 處理後音檔存放資料夾
train_audio_root = r'C:\Users\brad9\Desktop\BirdCLEF\train_audio'           # 原始音檔的根目錄（例如 'C:/.../train_audio'）

# === 載入CSV並篩選目標檔案 ===
df = pd.read_csv(csv_path)
target_df = df[(df['collection'] == 'CSA') & (df['rating'] == 0)]

# === 開始搬移檔案 ===
for idx, row in target_df.iterrows():
    relative_path = row['filename']   # 例如 112345/XC123456.ogg
    filename = os.path.basename(relative_path)  # 取出檔名本身
    processed_file = os.path.join(processed_folder, filename)
    target_path = os.path.join(train_audio_root, relative_path)

    if os.path.exists(processed_file):
        # 確保目標資料夾存在
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(processed_file, target_path)
        print(f'已覆蓋: {target_path}')
    else:
        print(f'[警告] 找不到處理後檔案: {processed_file}')

print('全部搬移完畢！')
