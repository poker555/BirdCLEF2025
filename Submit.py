import os
import json
import glob
import torch
import pandas as pd
import librosa
import timm
import numpy as np

# ================= 路徑設定 =================
soundscape_path = r"C:\Users\brad9\Desktop\BirdCLEF\train_soundscapes"
taxonomy_path = r"C:\Users\brad9\Desktop\BirdCLEF\taxonomy.csv"
logmel_path = r"C:\Users\brad9\Desktop\BirdCLEF\log_mel_v2.csv"
submit_csv_path = r"C:\Users\brad9\Desktop\BirdCLEF\submit.csv"
CONFIG_PATH = r"C:\Users\brad9\Desktop\BirdCLEF\audio_preprocessing_config.json"
model_path_dict = {
    "Insecta": r"C:\Users\brad9\Desktop\BirdCLEF\model\best_model_Insecta.pth",
    "Aves": r"C:\Users\brad9\Desktop\BirdCLEF\model\best_model_Aves.pth",
    "Amphibia": r"C:\Users\brad9\Desktop\BirdCLEF\model\best_model_Amphibia.pth",
    "Mammalia": r"C:\Users\brad9\Desktop\BirdCLEF\model\best_model_Mammalia.pth",
    "all": r"C:\Users\brad9\Desktop\BirdCLEF\model\best_model.pth",
}
num_classes_dict = {"Insecta": 16, "Aves": 146, "Amphibia": 34, "Mammalia": 9, "all": 4}
segment_sec = 5

# ================ 資料準備 =================
taxonomy_df = pd.read_csv(taxonomy_path)
logmel_df = pd.read_csv(logmel_path)
with open(CONFIG_PATH) as f:
    config = json.load(f)

# submit.csv 的所有欄位
submit_columns = ['row'] + taxonomy_df['primary_label'].tolist()

# ================ 載入模型 =================
def load_model(model_path, num_classes):
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes, in_chans=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model_all = load_model(model_path_dict["all"], 4)
model_dict = {
    group: load_model(model_path_dict[group], num_classes_dict[group])
    for group in ["Insecta", "Aves", "Amphibia", "Mammalia"]
}

# ================ 音訊切片前處理 ===============
def preprocess_segment(segment, sr, config):
    if config.get("apply_denoise", False):
        segment = librosa.effects.preemphasis(segment)
    rms = np.sqrt(np.mean(segment ** 2))
    if rms > 0:
        segment *= config["rms_target"] / (rms + 1e-9)
    mel_cfg = config["mel_spectrogram"]
    melspec = librosa.feature.melspectrogram(
        y=segment, sr=sr,
        n_fft=mel_cfg["n_fft"], hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"], fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"],
        power=mel_cfg["power"]
    )
    if config["log_scaling"]["apply_log_db"]:
        logmelspec = librosa.power_to_db(melspec, ref=np.max)
    else:
        logmelspec = melspec
    norm_cfg = config["normalization"]
    mean, std = np.mean(logmelspec), np.std(logmelspec)
    logmelspec = (logmelspec - mean) / (std + norm_cfg["epsilon"])
    return logmelspec

# ================== 推論與輸出 ==================
submit_rows = []
ogg_files = glob.glob(os.path.join(soundscape_path, '*.ogg'), recursive=True)
ogg_files = ogg_files[:10]  # 只取前 10 筆音檔進行測試

for file_path in ogg_files:
    y, sr = librosa.load(file_path, sr=None)
    total_sec = int(np.ceil(len(y) / sr))
    file_id = os.path.splitext(os.path.basename(file_path))[0]

    for start in range(0, total_sec, segment_sec):
        end = min(start + segment_sec, total_sec)
        segment = y[start*sr:end*sr]
        if len(segment) < sr:
            continue
        row_id = f"{file_id}_{end}"
        logmelspec = preprocess_segment(segment, sr, config)
        input_tensor = torch.tensor(logmelspec).unsqueeze(0).unsqueeze(0).float()

        new_row = dict.fromkeys(submit_columns, None)
        new_row['row'] = row_id

        with torch.no_grad():
            # 先用 model_all 判斷是哪個物種
            probs_all = torch.softmax(model_all(input_tensor), dim=1)
            idx = int(torch.argmax(probs_all))
            group = {0: "Insecta", 1: "Amphibia", 2: "Mammalia", 3: "Aves"}[idx]

            model = model_dict[group]
            class_col = f'class_name_{group}'
            class_map = logmel_df[[class_col, 'folder_id']].drop_duplicates()
            idx_to_folderid = dict(zip(class_map[class_col], class_map['folder_id']))
            probs = torch.softmax(model(input_tensor), dim=1).squeeze().cpu().numpy()

            for i in range(len(probs)):
                folder_id = idx_to_folderid.get(i)
                if folder_id in new_row:
                    new_row[folder_id] = float(probs[i])
        submit_rows.append(new_row)

submit_csv = pd.DataFrame(submit_rows, columns=submit_columns)
submit_csv = submit_csv.fillna(0)
submit_csv.to_csv(submit_csv_path, index=False)
print("輸出完成！檔案已儲存：", submit_csv_path)
