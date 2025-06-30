import os
import json
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback

# === 載入設定檔 ===
def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

# === 音訊處理 ===
def process_audio_file(file_path, config):
    y, sr = librosa.load(file_path, sr=config["sample_rate"], mono=config["mono"])

    if config["apply_trim"]:
        y, _ = librosa.effects.trim(y, top_db=config["trim_top_db"])

    if config["apply_denoise"]:
        y = librosa.effects.preemphasis(y)

    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y *= config["rms_target"] / (rms + 1e-9)

    segment_length = int(config["sample_rate"] * config["segment_duration_sec"])
    total_segments = len(y) // segment_length

    segments = []
    for i in range(total_segments):
        seg = y[i * segment_length: (i + 1) * segment_length]
        segments.append(seg)

    return segments, sr

# === 頻譜圖轉換與正規化 ===
def audio_to_logmel(segment, sr, config):
    mel_cfg = config["mel_spectrogram"]
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg["fmin"],
        fmax=mel_cfg["fmax"],
        power=mel_cfg["power"]
    )

    if config["log_scaling"]["apply_log_db"]:
        mel = librosa.power_to_db(mel, ref=np.max if config["log_scaling"]["ref_power"] == "max" else 1.0)

    norm_cfg = config["normalization"]
    if norm_cfg["type"] == "per_image_zscore":
        mean = np.mean(mel)
        std = np.std(mel)
        mel = (mel - mean) / (std + norm_cfg["epsilon"])
    elif norm_cfg["type"] == "minmax":
        mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel) + norm_cfg["epsilon"])

    return mel

# === 主流程：整批處理與更新 CSV ===
def process_dataset_and_update_csv(csv_path, audio_base_dir, output_dir, output_csv_path, config_path):
    config = load_config(config_path)
    df = pd.read_csv(csv_path)
    updated_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_rel_path = row["filename"]
        audio_path = os.path.join(audio_base_dir, audio_rel_path)

        if not os.path.exists(audio_path):
            continue

        try:
            segments, sr = process_audio_file(audio_path, config)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            rel_dir = os.path.dirname(audio_rel_path)

            output_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(output_subdir, exist_ok=True)

            for i, seg in enumerate(segments):
                mel = audio_to_logmel(seg, sr, config)
                out_name = f"{base_name}_seg{i}.npy"
                out_path = os.path.join(output_subdir, out_name)
                np.save(out_path, mel)

                new_row = row.copy()
                new_row["mel_filename"] = os.path.join(rel_dir, out_name)
                updated_rows.append(new_row)

        except Exception as e:
            print(f"❌ Error: {audio_path} | {e}")
            traceback.print_exc()

    out_df = pd.DataFrame(updated_rows)
    out_df.to_csv(output_csv_path, index=False)
    print(f"✅ 完成處理 {len(updated_rows)} 個切片，儲存於 {output_csv_path}")

# === 主程式執行區段 ===
if __name__ == "__main__":
    process_dataset_and_update_csv(
        csv_path="C:/Users/brad9/Desktop/BirdCLEF/train.csv",
        audio_base_dir="C:/Users/brad9/Desktop/BirdCLEF/train_audio",
        output_dir="C:/Users/brad9/Desktop/BirdCLEF/log_mel_16000kHZ",
        output_csv_path="C:/Users/brad9/Desktop/BirdCLEF/updated_train_mel.csv",
        config_path="C:/Users/brad9/Desktop/BirdCLEF/audio_preprocessing_config.json"
    )
