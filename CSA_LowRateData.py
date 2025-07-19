import pandas as pd
import os
import shutil

# === 設定路徑 ===
csv_path = r'C:\Users\brad9\Desktop\BirdCLEF\train.csv'
audio_root = r'C:\Users\brad9\Desktop\BirdCLEF\train_audio'
target_dir = r'C:\Users\brad9\Desktop\BirdCLEF\CSA_LowRate'

# 建立目標資料夾
os.makedirs(target_dir, exist_ok=True)

# 讀取CSV並篩選
df = pd.read_csv(csv_path)
filtered = df[(df['collection'] == 'CSA') & (df['rating'] == 0)]
file_list = filtered['filename'].tolist()

for rel_path in file_list:
    src_path = os.path.join(audio_root, rel_path)
    dst_path = os.path.join(target_dir, os.path.basename(rel_path))
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"移動 {src_path} -> {dst_path}")
    else:
        print(f"找不到檔案: {src_path}")

print('搬移完成！')
