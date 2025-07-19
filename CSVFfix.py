import pandas as pd

# 強制 primary_label 欄位型別為字串
meta = pd.read_csv(
    r'C:\Users\brad9\Desktop\BirdCLEF\train_meta.csv',
    dtype={'primary_label': str}
)

all_labels = sorted(meta['primary_label'].unique())
label2idx = {lbl: idx for idx, lbl in enumerate(all_labels)}

def to_onehot(label):
    onehot = [0] * len(all_labels)
    onehot[label2idx[label]] = 1
    return str(onehot)

meta['onehot'] = meta['primary_label'].map(to_onehot)

onehot_cols = [col for col in meta.columns if col.startswith('onehot_')]
meta = meta.drop(columns=onehot_cols)

meta.to_csv(r'C:\Users\brad9\Desktop\BirdCLEF\train_meta_onehotstr.csv', index=False)
print('完成：產生 onehot 欄位字串的 train_meta_onehotstr.csv')
print('label順序如下：')
for idx, lbl in enumerate(all_labels):
    print(f'{idx}: {lbl}')
