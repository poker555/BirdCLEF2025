import os
import json
import librosa
import numpy as np
from tqdm import tqdm

# 1. 載入 config
config_path = r"C:\Users\brad9\Desktop\BirdCLEF\audio_preprocessing_config.json"
with open(config_path) as f:
    config = json.load(f)

audio_dir = r"C:\Users\brad9\Desktop\BirdCLEF\train_audio\1564122"
output_dir = r"C:\Users\brad9\Desktop\BirdCLEF\log_mel_32000kHZ\1564122"
os.makedirs(output_dir, exist_ok=True)

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
    total_len = len(y)
    segments = []
    # 正常分段
    for i in range(0, total_len, segment_length):
        seg = y[i: i + segment_length]
        if len(seg) < segment_length:
            seg = np.pad(seg, (0, segment_length - len(seg)))  # 補0到指定長度
        segments.append(seg)
    return segments, sr

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

audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.ogg', '.wav', '.mp3'))]
for audio_file in tqdm(audio_files):
    audio_path = os.path.join(audio_dir, audio_file)
    try:
        segments, sr = process_audio_file(audio_path, config)
        base_name = os.path.splitext(audio_file)[0]
        for i, seg in enumerate(segments):
            mel = audio_to_logmel(seg, sr, config)
            out_name = f"{base_name}_seg{i}.npy"
            out_path = os.path.join(output_dir, out_name)
            np.save(out_path, mel)
    except Exception as e:
        print(f"❌ Error processing {audio_file}: {e}")

print("✅ 單一物種資料夾頻譜圖轉換完成！")
