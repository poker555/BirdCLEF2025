# Preprocessing script for BirdCLEF 2025 dataset
import os
import ast
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import h5py

# Load metadata
train_df = pd.read_csv(r'C:\Users\brad9\Desktop\BirdCLEF\train.csv')
taxonomy_df = pd.read_csv(r'C:\Users\brad9\Desktop\BirdCLEF\taxonomy.csv', dtype=str)
train_path = r'C:\Users\brad9\Desktop\BirdCLEF\train_audio'

# Build label set and mapping
all_labels = set()
for _, row in train_df.iterrows():
    all_labels.add(str(row['primary_label']))
    for s in ast.literal_eval(row['secondary_labels']):
        if s:
            all_labels.add(str(s))
label_list = sorted(list(all_labels))
label_to_index = {lbl: idx for idx, lbl in enumerate(label_list)}
num_classes = len(label_list)

# Compute total duration per species
total_duration = {lbl: 0.0 for lbl in label_list}
for _, row in train_df.iterrows():
    path = os.path.join(train_path, row['filename'])
    try:
        f = sf.SoundFile(path)
        duration = len(f) / f.samplerate
        f.close()
    except:
        y, sr = librosa.load(path, sr=None)
        duration = len(y) / sr
    species = [str(row['primary_label'])] + [s for s in ast.literal_eval(row['secondary_labels']) if s]
    for sp in species:
        if sp in total_duration:
            total_duration[sp] += duration

# Initialize HDF5
h5f = h5py.File('birdclef_data.h5', 'w')
h5f.create_dataset('spectrogram', shape=(0, 128, 313), maxshape=(None, 128, 313), dtype='float32', compression='gzip')
h5f.create_dataset('primary_label', shape=(0,), maxshape=(None,), dtype='int16', compression='gzip')
h5f.create_dataset('multi_label', shape=(0, num_classes), maxshape=(None, num_classes), dtype='uint8', compression='gzip')
h5f.create_dataset('rating', shape=(0,), maxshape=(None,), dtype='float32', compression='gzip')
h5f.create_dataset('split', shape=(0,), maxshape=(None,), dtype='S5', compression='gzip')
h5f.create_dataset('weight', shape=(0,), maxshape=(None,), dtype='float32', compression='gzip')
h5f.create_dataset('filename', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), compression='gzip')

# Parameters
sr = 32000
seg_dur = 5.0
seg_len = int(sr * seg_dur)
n_fft = 1024
hop_length = 150

buffer = {k: [] for k in ['spectrogram','primary_label','multi_label','rating','split','weight','filename']}

for _, row in train_df.iterrows():
    y, _ = librosa.load(os.path.join(train_path, row['filename']), sr=sr)
    rating = float(row['rating']) if 'rating' in row else 0.0
    primary = str(row['primary_label'])
    secondary = [s for s in ast.literal_eval(row['secondary_labels']) if s]
    multi = np.zeros(num_classes, dtype=np.uint8)
    multi[label_to_index[primary]] = 1
    for s in secondary:
        if s in label_to_index:
            multi[label_to_index[s]] = 1
    split = 'train' if random.random() < 0.8 else 'val'
    weight = 1.0 / total_duration[primary] if total_duration[primary] > 0 else 0.0

    for start in range(0, len(y), seg_len):
        seg = y[start:start+seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        if mel_db.shape[1] > 313:
            mel_db = mel_db[:, :313]
        elif mel_db.shape[1] < 313:
            mel_db = np.pad(mel_db, ((0,0),(0,313 - mel_db.shape[1])), mode='constant')
        if rating >= 3.0:
            mel_db -= mel_db.mean()
            mel_db /= mel_db.std() + 1e-6

        buffer['spectrogram'].append(mel_db)
        buffer['primary_label'].append(label_to_index[primary])
        buffer['multi_label'].append(multi)
        buffer['rating'].append(rating)
        buffer['split'].append(split.encode())
        buffer['weight'].append(weight)
        buffer['filename'].append(row['filename'])

        if len(buffer['spectrogram']) >= 100:
            n = h5f['spectrogram'].shape[0]
            new_n = n + len(buffer['spectrogram'])
            for k in buffer:
                shape = (new_n,) + h5f[k].shape[1:] if len(h5f[k].shape) > 1 else (new_n,)
                h5f[k].resize(shape)
                h5f[k][n:new_n] = buffer[k]
            for k in buffer:
                buffer[k] = []

# write remaining
if buffer['spectrogram']:
    n = h5f['spectrogram'].shape[0]
    new_n = n + len(buffer['spectrogram'])
    for k in buffer:
        shape = (new_n,) + h5f[k].shape[1:] if len(h5f[k].shape) > 1 else (new_n,)
        h5f[k].resize(shape)
        h5f[k][n:new_n] = buffer[k]
    for k in buffer:
        buffer[k] = []

h5f.close()
