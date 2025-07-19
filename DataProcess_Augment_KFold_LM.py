import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import h5py
from tqdm import tqdm
import multiprocessing as mp
import json

# ====== 設定 ======
AUDIO_DIR = r"C:\Users\brad9\Desktop\BirdCLEF\train_audio"
META_CSV = r"C:\Users\brad9\Desktop\BirdCLEF\code\meta_all.csv"
CONFIG_PATH = r"C:\Users\brad9\Desktop\BirdCLEF\config.json"
OUTPUT_H5 = r"C:\Users\brad9\Desktop\BirdCLEF\train_all_withAugment.h5"
N_PROCESS = max(1, mp.cpu_count() // 2)

# ====== 讀設定檔 ======
with open(CONFIG_PATH) as f:
    config = json.load(f)

segment_sec = config.get("segment_sec", 5)
sample_rate = config.get("sample_rate", 32000)
n_mels = config.get("n_mels", 128)
hop_length = config.get("hop_length", 512)

# ====== 讀 meta 檔 ======
meta_df = pd.read_csv(META_CSV)
label_list = sorted(meta_df['primary_label'].unique())
label2idx = {k: i for i, k in enumerate(label_list)}
mel_len = int(np.ceil(segment_sec * sample_rate / hop_length))

def apply_augment(y, sr, level_cfg):
    # Pitch shift
    if "pitch_shift" in level_cfg and level_cfg["pitch_shift"] != [0,0]:
        if np.random.rand() < 0.5:
            min_s, max_s = level_cfg["pitch_shift"]
            n_steps = np.random.uniform(min_s, max_s)
            y = librosa.effects.pitch_shift(y, sr, n_steps)
    # Time stretch
    if "time_stretch" in level_cfg and level_cfg["time_stretch"] != [1.0,1.0]:
        if np.random.rand() < 0.5:
            min_r, max_r = level_cfg["time_stretch"]
            rate = np.random.uniform(min_r, max_r)
            y = librosa.effects.time_stretch(y, rate)
            if len(y) < segment_sec * sr:
                y = np.pad(y, (0, int(segment_sec*sr)-len(y)))
            else:
                y = y[:int(segment_sec*sr)]
    # Volume
    if "volume" in level_cfg and level_cfg["volume"] != [1.0,1.0]:
        if np.random.rand() < 0.5:
            min_v, max_v = level_cfg["volume"]
            scale = np.random.uniform(min_v, max_v)
            y = y * scale
    # Noise
    if level_cfg.get("add_noise", False):
        if np.random.rand() < 0.5:
            snr_db = level_cfg.get("snr_db", 22)
            rms = np.sqrt(np.mean(y**2))
            noise_std = rms / (10**(snr_db/20))
            noise = np.random.normal(0, noise_std, size=y.shape)
            y = y + noise
    return y

def process_row(row):
    audio_path = os.path.join(AUDIO_DIR, row['filename'])
    seg_idx = int(row['segment_id'].split('_seg')[-1])
    seg_start = seg_idx * segment_sec * sample_rate
    seg_end = seg_start + segment_sec * sample_rate
    y = np.zeros(segment_sec * sample_rate, dtype=np.float32)
    try:
        wav, sr = sf.read(audio_path, dtype='float32')
        if sr != sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        # 切 segment
        y = wav[int(seg_start):int(seg_end)]
        if len(y) < segment_sec * sample_rate:
            y = np.pad(y, (0, segment_sec*sample_rate - len(y)))
        else:
            y = y[:segment_sec*sample_rate]
    except Exception as e:
        print(f"讀取失敗 {audio_path}: {e}")
        return None

    # only train 做增強
    if row['split'] == 'train':
        lvl = row['augment_level']
        y = apply_augment(y, sample_rate, config[lvl])

    mel = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_mels=n_mels, hop_length=hop_length, power=2.0  # <--- y=!!
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape[1] < mel_len:
        mel = np.pad(mel, ((0,0), (0, mel_len-mel.shape[1])), 'constant')
    elif mel.shape[1] > mel_len:
        mel = mel[:, :mel_len]

    label = np.array(list(map(int, row['onehot'].split(','))), dtype=np.float32)
    return (
        mel.astype(np.float32),
        label,
        row['split'],
        row['segment_id'],
        row['augment_level'],
        row['fold']
    )

def process_all(meta_df):
    rows = meta_df.to_dict('records')
    with mp.Pool(N_PROCESS) as pool:
        for i, out in enumerate(pool.imap(process_row, rows)):
            yield i, out

if __name__ == "__main__":  # 必須要這一行！
    with h5py.File(OUTPUT_H5, "w") as h5f:
        N = len(meta_df)
        h5f.create_dataset("mel", shape=(N, n_mels, mel_len), dtype=np.float32)
        h5f.create_dataset("label", shape=(N, len(label_list)), dtype=np.float32)
        h5f.create_dataset("split", shape=(N,), dtype='S10')
        h5f.create_dataset("segment_id", shape=(N,), dtype=h5py.string_dtype())
        h5f.create_dataset("augment_level", shape=(N,), dtype='S1')
        h5f.create_dataset("fold", shape=(N,), dtype='i')
        for i, out in tqdm(process_all(meta_df), total=len(meta_df), desc="mel生成"):
            if out is None:
                continue
            mel, label, split, segid, lvl, fold = out
            h5f["mel"][i] = mel
            h5f["label"][i] = label
            h5f["split"][i] = split.encode()
            h5f["segment_id"][i] = segid
            h5f["augment_level"][i] = lvl.encode()
            h5f["fold"][i] = int(fold)

    print(f"HDF5 生成完畢：{OUTPUT_H5}")
