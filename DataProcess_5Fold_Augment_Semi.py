import os
import json
import torchaudio
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import h5py
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# ====== 讀取設定 ======
CONFIG_PATH = r'C:\Users\brad9\Desktop\BirdCLEF\config_dataProcess.json'
TRAIN_CSV = r'C:\Users\brad9\Desktop\BirdCLEF\train.csv'
AUDIO_ROOT = r'C:\Users\brad9\Desktop\BirdCLEF\train_audio'
H5_SAVE_PATH = r'C:\Users\brad9\Desktop\BirdCLEF\train_valid_weighted.h5'
MAX_WORKERS = 8

with open(CONFIG_PATH) as f:
    config = json.load(f)
pre_cfg = config['preprocessing']
aug_cfg = pre_cfg.get("augmentations", {})

# ====== 資料讀取 ======
df = pd.read_csv(TRAIN_CSV)

# ====== 統計每個物種的總秒數 ======
label_duration = defaultdict(float)
for _, row in df.iterrows():
    try:
        path = os.path.join(AUDIO_ROOT, row['filename'])
        dur = sf.info(path).duration
        label_duration[row['primary_label']] += dur
    except:
        print(f"[警告] 無法讀取 {row['filename']}")

# ====== 計算 label → idx 映射 ======
all_labels = sorted(df['primary_label'].unique())
label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
num_class = len(all_labels)

# ====== 分割 train/valid（保證音訊不跨集） ======
filename_to_label = df.drop_duplicates('filename').set_index('filename')['primary_label'].to_dict()
all_filenames = list(filename_to_label.keys())
train_files, valid_files = train_test_split(
    all_filenames, test_size=0.2, stratify=[filename_to_label[f] for f in all_filenames], random_state=42
)
filename_to_split = {fn: 'train' for fn in train_files}
filename_to_split.update({fn: 'valid' for fn in valid_files})

# ====== 增強工具 ======
def apply_pitch_shift(y, sr, cfg):
    import librosa
    n_steps = np.random.uniform(cfg['min_semitones'], cfg['max_semitones'])
    return librosa.effects.pitch_shift(y, sr, n_steps)

def apply_time_stretch(y, cfg):
    import librosa
    rate = np.random.uniform(cfg['min_rate'], cfg['max_rate'])
    y_stretch = librosa.effects.time_stretch(y, rate)
    return librosa.util.fix_length(y_stretch, len(y))

def apply_volume_gain(y, cfg):
    gain_db = np.random.uniform(*cfg['gain_db_range'])
    return y * (10 ** (gain_db / 20))

def choose_augmentations(rating):
    # 動態策略選擇
    if rating >= 4:
        return ['pitch_shift', 'time_stretch', 'volume_gain']
    elif rating >= 2:
        return ['pitch_shift', 'volume_gain']
    else:
        return ['volume_gain']

# ====== 音訊轉 mel ======
def audio_to_mel(y, sr, mel_cfg):
    melspec_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=mel_cfg['n_fft'],
        hop_length=mel_cfg['hop_length'],
        n_mels=mel_cfg['n_mels'],
        f_min=mel_cfg['fmin'],
        f_max=mel_cfg['fmax'],
        power=mel_cfg.get('power', 2.0)
    )
    mel = melspec_fn(torch.tensor(y).unsqueeze(0)).squeeze(0).numpy()
    return mel

# ====== 主處理函式 ======
def process_one(row):
    filename = row['filename']
    label = row['primary_label']
    label_idx = label_to_idx[label]
    sr = pre_cfg['sample_rate']
    split = filename_to_split.get(filename, 'unknown')
    rating = row['rating']
    weight = 1.0 / label_duration[label] if label_duration[label] > 0 else 1.0

    # 根據秒數決定複製倍數
    dur = label_duration[label]
    if dur < 100:
        factor = pre_cfg['augmentation_factor_by_duration']['less_than_100']
    elif dur < 1000:
        factor = pre_cfg['augmentation_factor_by_duration']['between_100_1000']
    else:
        factor = pre_cfg['augmentation_factor_by_duration']['more_than_1000']

    path = os.path.join(AUDIO_ROOT, filename)
    try:
        y, orig_sr = torchaudio.load(path)
        y = y.mean(0).numpy()
        if orig_sr != sr:
            y = torchaudio.transforms.Resample(orig_sr, sr)(torch.tensor(y)).numpy()

        if pre_cfg.get('apply_trim', False):
            import librosa
            y, _ = librosa.effects.trim(y, top_db=pre_cfg.get('trim_top_db', 25))

        if pre_cfg.get('rms_target') is not None:
            rms = np.sqrt(np.mean(y**2))
            if rms > 0:
                y *= pre_cfg['rms_target'] / rms

        seg_len = int(pre_cfg['segment_duration_sec'] * sr)
        results = []

        for _ in range(factor):
            start = 0
            if len(y) > seg_len:
                start = np.random.randint(0, len(y) - seg_len)
            seg = y[start:start + seg_len]
            if len(seg) < seg_len:
                seg = np.pad(seg, (0, seg_len - len(seg)))

            # 動態增強
            aug_list = choose_augmentations(rating)
            for aug in aug_list:
                cfg = aug_cfg.get(aug, {})
                if cfg.get('apply') and np.random.rand() < cfg.get('p', 1.0):
                    if aug == 'pitch_shift':
                        seg = apply_pitch_shift(seg, sr, cfg)
                    elif aug == 'time_stretch':
                        seg = apply_time_stretch(seg, cfg)
                    elif aug == 'volume_gain':
                        seg = apply_volume_gain(seg, cfg)

            mel = audio_to_mel(seg, sr, pre_cfg['mel_spectrogram'])
            if pre_cfg.get('log_scaling', {}).get('apply_log_db', True):
                mel = 10 * np.log10(np.maximum(mel, 1e-10))
            if pre_cfg.get('normalization', {}).get('type') == 'per_image_zscore':
                mel = (mel - mel.mean()) / (mel.std() + pre_cfg['normalization'].get('epsilon', 1e-6))

            results.append({
                'mel': mel.astype(np.float32),
                'label_idx': label_idx,
                'filename': filename,
                'quality': rating,
                'split': split,
                'weight': weight
            })

        return results
    except Exception as e:
        print(f"[錯誤] 處理 {filename} 失敗: {e}")
        return []

# ====== 執行處理 ======
tasks = df.drop_duplicates('filename').to_dict(orient='records')
results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    for r in tqdm(ex.map(process_one, tasks), total=len(tasks)):
        results.extend(r)

# ====== 整理 HDF5 ======
print(f"總共寫入 {len(results)} 筆資料")

mels = np.stack([r['mel'] for r in results])
label_ids = np.array([r['label_idx'] for r in results], dtype='int32')
onehots = np.zeros((len(results), num_class), dtype='int8')
for i, idx in enumerate(label_ids):
    onehots[i, idx] = 1
qualities = np.array([r['quality'] for r in results], dtype='int8')
filenames = np.array([r['filename'] for r in results], dtype='S64')
splits = np.array([r['split'] for r in results], dtype='S16')
weights = np.array([r['weight'] for r in results], dtype='float32')

with h5py.File(H5_SAVE_PATH, 'w') as h5f:
    h5f.create_dataset('mels', data=mels, dtype='float32')
    h5f.create_dataset('label_idx', data=label_ids)
    h5f.create_dataset('onehot', data=onehots)
    h5f.create_dataset('quality', data=qualities)
    h5f.create_dataset('filename', data=filenames)
    h5f.create_dataset('split', data=splits)
    h5f.create_dataset('weight', data=weights)

print(f"[完成] HDF5 儲存於：{H5_SAVE_PATH}")
