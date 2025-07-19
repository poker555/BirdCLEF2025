import os
import pandas as pd
import numpy as np
import soundfile as sf
import h5py
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

def process_row(args):
    idx, row_dict, mel_shape1, N_MELS, SAMPLE_RATE, TARGET_LEN, HOP_LENGTH, FILE_BASE = args
    import librosa
    import numpy as np
    import os
    audio_path = os.path.join(FILE_BASE, row_dict['filename'])
    seg = int(row_dict['segment_id'].split('_seg')[-1])
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"讀取失敗：{audio_path}, {e}")
        return idx, None, None
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
        slices = [y]
    else:
        slices = [y[i:i+TARGET_LEN] for i in range(0, len(y)-TARGET_LEN+1, TARGET_LEN)]
    y_slice = slices[seg]
    mel = librosa.feature.melspectrogram(
        y=y_slice, sr=SAMPLE_RATE,
        n_mels=N_MELS, hop_length=HOP_LENGTH, power=2.0
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape[1] < mel_shape1:
        pad_width = mel_shape1 - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad_width)), mode='constant')
    elif mel.shape[1] > mel_shape1:
        mel = mel[:, :mel_shape1]
    label_arr = np.array(list(map(int, row_dict["onehot"].split(","))), dtype=np.float32)
    return idx, mel.astype(np.float32), label_arr

if __name__ == "__main__":
    # ====== 基本設定 ======
    FILE_BASE = r"C:\Users\brad9\Desktop\BirdCLEF++\train_audio"
    CSV_PATH = r"C:\Users\brad9\Desktop\BirdCLEF++\train.csv"
    OUTPUT_H5 = r"C:\Users\brad9\Desktop\BirdCLEF++\train_v2.h5"
    OUTPUT_CSV = r"C:\Users\brad9\Desktop\BirdCLEF++\train_v2.csv"
    TARGET_DURATION = 5  # 秒
    SAMPLE_RATE = 32000
    TARGET_LEN = TARGET_DURATION * SAMPLE_RATE
    N_MELS = 128
    HOP_LENGTH = 512
    MAX_THREADS = 8  # 依主機核心數調整

    # ====== 1. 讀取資料和標籤編碼 ======
    df = pd.read_csv(CSV_PATH)
    all_labels = set()
    for label_str in df['primary_label']:
        all_labels.update([l.strip() for l in str(label_str).split(',')])
    all_labels = sorted(list(all_labels))
    label2idx = {label: idx for idx, label in enumerate(all_labels)}

    def encode_labels(label_str):
        onehot = np.zeros(len(all_labels), dtype=int)
        if pd.isna(label_str): return onehot
        for l in str(label_str).split(','):
            l = l.strip()
            if l in label2idx:
                onehot[label2idx[l]] = 1
        return onehot

    # ====== 2. 多執行緒取得音檔長度 ======
    def get_audio_length(row):
        audio_path = os.path.join(FILE_BASE, row['filename'])
        try:
            info = sf.info(audio_path)
            return row['filename'], info.frames
        except Exception as e:
            print(f"讀取失敗：{audio_path}, {e}")
            return row['filename'], 0

    audio_lengths = {}
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(get_audio_length, row) for _, row in df.iterrows()]
        for fut in tqdm(as_completed(futures), total=len(df), desc="多執行緒讀音檔長度"):
            fn, length = fut.result()
            audio_lengths[fn] = length

    # ====== 3. 計算全部切片數 ======
    total_slices = 0
    slices_dict = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="計算切片數"):
        audio_len = audio_lengths[row['filename']]
        if audio_len == 0:
            slices = 0
        elif audio_len < TARGET_LEN:
            slices = 1
        else:
            slices = int(np.floor((audio_len-TARGET_LEN)/TARGET_LEN+1))
            slices = max(slices, 1)
        slices_dict[row['filename']] = slices
        total_slices += slices
    print(f"全部切片數: {total_slices}")

    # ====== 4. 分割 train/valid 與 meta（onehot存同一欄） ======
    meta_records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="資料分割與meta紀錄"):
        audio_len = audio_lengths[row['filename']]
        slices = slices_dict[row['filename']]
        if slices == 0:
            continue
        onehot = encode_labels(row['primary_label'])
        onehot_str = ','.join(map(str, onehot))  # 例如 '0,1,1,0,0,0'
        for i in range(slices):
            meta_records.append({
                "filename": row["filename"],
                "primary_label": row["primary_label"],
                "segment_id": f"{row['filename']}_seg{i}",
                "onehot": onehot_str
            })

    new_df = pd.DataFrame(meta_records)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    groups = new_df['filename']
    train_idx, valid_idx = next(gss.split(new_df, groups=groups))
    new_df = new_df.reset_index(drop=True)
    new_df['split'] = "none"
    new_df.loc[train_idx, 'split'] = 'train'
    new_df.loc[valid_idx, 'split'] = 'valid'
    new_df.to_csv(OUTPUT_CSV, index=False)

    # ====== 5. Mel 平行處理與 HDF5 寫入 ======
    mel_shape1 = int(np.ceil(TARGET_LEN / HOP_LENGTH))
    args_list = [
        (idx, row.to_dict(), mel_shape1, N_MELS, SAMPLE_RATE, TARGET_LEN, HOP_LENGTH, FILE_BASE)
        for idx, row in new_df.iterrows()
    ]
    with h5py.File(OUTPUT_H5, "w") as h5f:
        mel_ds = h5f.create_dataset("mel", shape=(len(new_df), N_MELS, mel_shape1), dtype=np.float32)
        label_ds = h5f.create_dataset("label", shape=(len(new_df), len(all_labels)), dtype=np.float32)
        split_ds = h5f.create_dataset("split", shape=(len(new_df),), dtype='S5')
        segid_ds = h5f.create_dataset("segment_id", shape=(len(new_df),), dtype=h5py.string_dtype())
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for idx, mel, label in tqdm(pool.imap_unordered(process_row, args_list), total=len(args_list), desc="平行處理寫入HDF5"):
                if mel is None:
                    continue
                mel_ds[idx] = mel
                label_ds[idx] = label
                split_ds[idx] = new_df.loc[idx, 'split'].encode()
                segid_ds[idx] = new_df.loc[idx, 'segment_id']
        print(f"已存入 HDF5：{OUTPUT_H5}")
