# ========== 程式碼一：分類四個物種 (Amphibia, Aves, Insecta, Mammalia) ==========

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from timm import create_model
import torchaudio.transforms as T

# ==== 參數設定 ====
CSV_PATH = "C:/Users/brad9/Desktop/BirdCLEF/updated_train_mel.csv"
MEL_DIR = "C:/Users/brad9/Desktop/BirdCLEF/log_mel"
CONFIG_PATH = "C:/Users/brad9/Desktop/BirdCLEF/audio_preprocessing_config.json"
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 載入 species_class_to_idx 與 config ====
with open("C:/Users/brad9/Desktop/BirdCLEF/index/species_class_to_idx.json") as f:
    species_class_to_idx = json.load(f)

with open(CONFIG_PATH) as f:
    config = json.load(f)

# ==== Dataset ====
class LogMelSpeciesDataset(Dataset):
    def __init__(self, df, mel_dir, class_map, config, transform=None):
        self.df = df
        self.mel_dir = mel_dir
        self.class_map = class_map
        self.transform = transform
        self.config = config
        self.num_classes = len(class_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_path = os.path.join(self.mel_dir, row["mel_filename"])
        mel = np.load(mel_path)
        mel = (mel - mel.mean()) / (mel.std() + self.config["normalization"].get("epsilon", 1e-6))
        mel = np.expand_dims(mel, axis=0)
        mel = torch.tensor(mel, dtype=torch.float32)

        if self.transform:
            mel = self.transform(mel)

        label_idx = self.class_map[row["species_class"]]
        label = torch.zeros(self.num_classes)
        label[label_idx] = 1.0
        return mel, label

# ==== 數據增強 ====
def get_augment(config):
    aug_cfg = config.get("augmentation", {})
    strat_cfg = config.get("augmentation_strategy", {})

    def augment(x):
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])

        if strat_cfg.get("specaugment", {}).get("apply", False):
            x = T.FrequencyMasking(freq_mask_param=strat_cfg["specaugment"].get("frequency_mask_param", 12))(x)
            x = T.TimeMasking(time_mask_param=strat_cfg["specaugment"].get("time_mask_param", 24))(x)

        if strat_cfg.get("additive_noise", {}).get("apply", False):
            snr = strat_cfg["additive_noise"].get("snr_db", 20)
            noise = torch.randn_like(x) * (10 ** (-snr / 20))
            x = x + noise

        if strat_cfg.get("random_time_crop", {}).get("apply", False):
            ratio = strat_cfg["random_time_crop"].get("crop_ratio", 0.9)
            t = x.shape[2]
            crop_len = int(t * ratio)
            if crop_len < t:
                start = torch.randint(0, t - crop_len + 1, (1,)).item()
                x = x[:, :, start:start + crop_len]

        return x

    return augment

# ==== 訓練 ====
def train():
    # 載入 taxonomy.csv 對應 species_class
    taxonomy = pd.read_csv("C:/Users/brad9/Desktop/BirdCLEF/taxonomy.csv")
    taxonomy.columns = taxonomy.columns.str.strip()
    taxonomy.columns = [col.replace("﻿", "") for col in taxonomy.columns]
    taxonomy = taxonomy[["primary_label", "class_name"]].drop_duplicates()

    df = pd.read_csv(CSV_PATH)
    df = df.merge(taxonomy, left_on="primary_label", right_on="primary_label", how="left")
    df.rename(columns={"class_name": "species_class"}, inplace=True)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = LogMelSpeciesDataset(train_df, MEL_DIR, species_class_to_idx, config, transform=get_augment(config))
    val_ds = LogMelSpeciesDataset(val_df, MEL_DIR, species_class_to_idx, config)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = create_model("efficientnet_b3", pretrained=True, in_chans=1, num_classes=len(species_class_to_idx)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):
        print("=" * 60)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0
        from tqdm import tqdm
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

        # 驗證階段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

                pred = torch.sigmoid(out) > 0.5
                correct += (pred == y.bool()).all(dim=1).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total * 100
        print(f"Training Loss: {avg_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")
        scheduler.step(avg_val_loss)

        # 模型儲存與 early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✓ Model saved (new best validation loss)")
        else:
            patience_counter += 1
            print(f"✗ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train()
