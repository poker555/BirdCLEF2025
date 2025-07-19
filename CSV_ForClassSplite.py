import pandas as pd

# 讀取你的meta
meta = pd.read_csv(r'C:\Users\brad9\Desktop\BirdCLEF\code\meta_all.csv')

N_FOLDS = meta['fold'].nunique() if meta['fold'].nunique() > 1 else 3
all_species = set(meta['class_name'].unique())

# 對每個fold檢查物種
for fold in sorted(meta['fold'].unique()):
    fold_df = meta[meta['fold'] == fold]
    fold_species = set(fold_df['class_name'])
    missing = all_species - fold_species
    if missing:
        print(f"Fold {fold} 缺少物種：{missing}")
        # 將這些物種的音檔強制分配到訓練集（fold=-2）
        for specie in missing:
            # 找到這個物種所有相關的filename
            missing_files = set(meta[meta['class_name'] == specie]['filename'])
            meta.loc[meta['filename'].isin(missing_files), 'fold'] = -2  # -2代表只能進train

# split標記（-2 fold一律train）
meta['split'] = 'train'
for fold in sorted(meta['fold'].unique()):
    if fold >= 0:
        meta.loc[meta['fold'] == fold, 'split'] = 'valid'
meta.loc[meta['fold'] == -2, 'split'] = 'train'

# 儲存新檔案
meta.to_csv('meta_all_fix.csv', index=False)
print("修正完成！新檔案：meta_all_fix.csv")
