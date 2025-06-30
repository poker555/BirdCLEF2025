import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split

class AudioWaveformDataset(Dataset):
    def __init__(self, df, audio_dir, mask, class_map, config, duration=5.0):
        self.df = df
        self.audio_dir = audio_dir
        self.mask = mask
        self.class_map = class_map
        self.config = config
        self.duration = duration
        self.id_map = {self.class_map[label]: i for i, label in enumerate(mask)}
        self.sample_rate = config.get("sample_rate", 16000)
        self.segment_length = int(self.sample_rate * self.duration)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["audio_filename"])
        # 讀音訊且保證長度
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        # 靜音裁切（可根據config控制）
        if self.config.get("apply_trim", False):
            y, _ = librosa.effects.trim(y, top_db=self.config.get("trim_top_db", 30))
        # RMS normalize
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 0:
            y *= self.config.get("rms_target", 0.1) / (rms + 1e-9)
        # segment 長度固定
        if len(y) < self.segment_length:
            y = np.pad(y, (0, self.segment_length - len(y)), mode='constant')
        else:
            y = y[:self.segment_length]
        # 可以在這裡加入更多 augmentation（如 pitch shift, time stretch, noise, specaugment ...）

        # 處理標籤（多標籤 one-hot）
        label = torch.zeros(len(self.mask), dtype=torch.float32)
        if "multi_labels" in row.index and not pd.isnull(row["multi_labels"]):
            label_names = str(row["multi_labels"]).split(",")
            for name in label_names:
                name = name.strip()
                if name in self.class_map and self.class_map[name] in self.id_map:
                    label_idx = self.id_map[self.class_map[name]]
                    label[label_idx] = 1.0
        else:
            original = str(row["primary_label"])
            if original in self.class_map and self.class_map[original] in self.id_map:
                label_idx = self.id_map[self.class_map[original]]
                label[label_idx] = 1.0

        return torch.from_numpy(y).float(), label

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# 路徑與參數設定
AUDIO_DIR = "C:/Users/brad9/Desktop/BirdCLEF/original_audio_16kHz"
CSV_PATH = "C:/Users/brad9/Desktop/BirdCLEF/your_audio_csv.csv"
MASK_PATHS = { ... }  # 和你原本一樣
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "C:/Users/brad9/Desktop/BirdCLEF/audio_preprocessing_config.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)
with open("C:/Users/brad9/Desktop/BirdCLEF/index/organism_class_to_idx.json") as f:
    organism_class_to_idx = json.load(f)
with open("C:/Users/brad9/Desktop/BirdCLEF/index/idx_to_organism_class.json") as f:
    idx_to_organism = json.load(f)

def train(organism_type):
    assert organism_type in MASK_PATHS, f"不支援的類別: {organism_type}"
    with open(MASK_PATHS[organism_type]) as f:
        class_mask_idx = json.load(f)
    class_mask = [idx_to_organism[str(i)] for i in class_mask_idx]
    df = pd.read_csv(CSV_PATH)
    df = df[df["primary_label"].isin(class_mask)]
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = AudioWaveformDataset(train_df, AUDIO_DIR, class_mask, organism_class_to_idx, config)
    val_ds = AudioWaveformDataset(val_df, AUDIO_DIR, class_mask, organism_class_to_idx, config)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    feature_extractor = ASTFeatureExtractor()
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=len(class_mask),
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for x, y in loop:
            # x: [B, 音訊長度]
            x_np = [xx.cpu().numpy() for xx in x]
            inputs = feature_extractor(
                x_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs['input_values'].to(DEVICE)
            y = y.to(DEVICE)
            outputs = model(input_values=input_values, return_dict=True)
            logits = outputs.logits
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"[{organism_type}] Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating", leave=False):
                x_np = [xx.cpu().numpy() for xx in x]
                inputs = feature_extractor(
                    x_np,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                input_values = inputs['input_values'].to(DEVICE)
                y = y.to(DEVICE)
                outputs = model(input_values=input_values, return_dict=True)
                logits = outputs.logits
                loss = criterion(logits, y)
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(y.cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        binary_preds = (preds > 0.5).astype(int)
        acc = (binary_preds == labels).mean()
        print(f"[{organism_type}] Epoch {epoch+1}] Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train("Aves")
