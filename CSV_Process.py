import pandas as pd
import ast
import numpy as np

# 檔案路徑（請根據你的實際路徑調整）
meta_path = r"C:\Users\brad9\Desktop\BirdCLEF\train_meta.csv"
augment_path = r"C:\Users\brad9\Desktop\BirdCLEF\train_Augment_KFold_Semi.csv"
output_path = r"C:\Users\brad9\Desktop\BirdCLEF\train_meta_v2.csv"

# 1. 讀取 meta 檔，並整理 onehot 欄位
meta = pd.read_csv(meta_path, dtype={'primary_label': str})

# 找出所有 onehot_ 開頭欄位
onehot_cols = [c for c in meta.columns if c.startswith("onehot_")]
# 將多個 onehot 欄位合併成一欄字串，格式為 '[0,1,0,...]'
meta['onehot'] = meta[onehot_cols].values.tolist()
meta['onehot'] = meta['onehot'].apply(lambda x: str(list(map(int, x))))
# 刪除舊 onehot_XXX 欄
meta = meta.drop(columns=onehot_cols)

# 2. 讀取並合併 total_seconds
augment = pd.read_csv(augment_path, dtype={'primary_label': str})
# 若有重複primary_label, total_seconds只取一筆
seconds_map = augment.drop_duplicates('primary_label').set_index('primary_label')['total_seconds'].to_dict()
meta['total_seconds'] = meta['primary_label'].map(seconds_map)

# 3. 建立加權欄位
# 合理加權常用方式為：weight = 1 / (total_seconds + epsilon)
epsilon = 1.0   # 避免除以零
meta['weight'] = 1.0 / (meta['total_seconds'] + epsilon)
# 為了避免個別權重極大，可以做正規化（如使平均為1）
meta['weight'] = meta['weight'] / meta['weight'].mean()

# 輸出
meta.to_csv(output_path, index=False)
print(f"已產生：{output_path}")
