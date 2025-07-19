import os
import torch
import timm
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# 參數區
SOUNDSCAPE_DIR = r"C:\Users\brad9\Desktop\BirdCLEF++\train_soundscapes"
MODEL_PATH = r"C:\Users\brad9\Desktop\BirdCLEF++\best_model_v2.pth"
OUTPUT_CSV = r"C:\Users\brad9\Desktop\BirdCLEF++\submission.csv"
SAMPLE_SUB_CSV = r"C:\Users\brad9\Desktop\BirdCLEF++\sample_submission.csv"
IMG_SIZE = 300
SAMPLE_RATE = 32000
TARGET_LEN = 5 * SAMPLE_RATE
N_MELS = 128
HOP_LENGTH = 512

# === 讀取 submission 欄位/物種 ===
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
target_cols = [c for c in sample_sub.columns if c != 'row_id']

# 取得全部 ogg 路徑
# #audio_files = [os.path.join(SOUNDSCAPE_DIR, f) for f in os.listdir(SOUNDSCAPE_DIR) if f.endswith('.ogg')]
audio_files = [os.path.join(SOUNDSCAPE_DIR, f) for f in os.listdir(SOUNDSCAPE_DIR) if f.endswith('.ogg')][:10]

# 建立 EfficientNetB3（跟訓練一致）
def get_model(num_classes):
    model = timm.create_model('efficientnet_b3', pretrained=True, in_chans=3)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(target_cols)
model = get_model(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# == 預測主流程 ==
results = []
for file in tqdm(audio_files, desc="預測 soundscape"):
    y, sr = librosa.load(file, sr=SAMPLE_RATE)
    if len(y) < 60 * SAMPLE_RATE:
        y = np.pad(y, (0, 60 * SAMPLE_RATE - len(y)))
    basename = os.path.splitext(os.path.basename(file))[0]
    # 切片（每 5 秒，共 12 片）
    for seg_idx, t_start in enumerate(range(0, 60, 5)):
        start_sample = t_start * SAMPLE_RATE
        end_sample = start_sample + TARGET_LEN
        y_slice = y[start_sample:end_sample]
        if len(y_slice) < TARGET_LEN:
            y_slice = np.pad(y_slice, (0, TARGET_LEN - len(y_slice)))
        # Mel 頻譜
        mel = librosa.feature.melspectrogram(
            y=y_slice, sr=SAMPLE_RATE,
            n_mels=N_MELS, hop_length=HOP_LENGTH, power=2.0
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0).repeat(3, 1, 1)
        mel = torch.nn.functional.interpolate(
            mel.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
        ).squeeze(0).unsqueeze(0)
        mel = mel.to(device)
        # 預測
        with torch.no_grad():
            logits = model(mel)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        # row_id 用結束秒數
        row_id = f"{basename}_{t_start+5}"
        row = {'row_id': row_id}
        for i, col in enumerate(target_cols):
            row[col] = probs[i]
        results.append(row)

# 匯出 CSV，欄位與 sample_submission 一致
out_df = pd.DataFrame(results)
# 若有缺欄位，自動補0（通常不會發生）
for c in target_cols:
    if c not in out_df:
        out_df[c] = 0.0
# 保持欄位順序
out_df = out_df[['row_id'] + target_cols]
out_df.to_csv(OUTPUT_CSV, index=False, float_format="%.8f")
print(f"已產生 submission: {OUTPUT_CSV}")
