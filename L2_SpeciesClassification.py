import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm
from torch.amp import GradScaler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score
import random

# ==== 基本參數 ====
MASK_PATHS = {
    "Amphibia": "C:/Users/brad9/Desktop/BirdCLEF/index/organism_masks/class_mask_species_Amphibia.json",
    "Aves": "C:/Users/brad9/Desktop/BirdCLEF/index/organism_masks/class_mask_species_Aves.json",
    "Insecta": "C:/Users/brad9/Desktop/BirdCLEF/index/organism_masks/class_mask_species_Insecta.json",
    "Mammalia": "C:/Users/brad9/Desktop/BirdCLEF/index/organism_masks/class_mask_species_Mammalia.json",
}
TO_Organism = "C:/Users/brad9/Desktop/BirdCLEF/index/idx_to_organism_class.json"
TO_IDX = "C:/Users/brad9/Desktop/BirdCLEF/index/organism_class_to_idx.json"
CSV_PATH = "C:/Users/brad9/Desktop/BirdCLEF/updated_train_mel.csv"
MEL_DIR = "C:/Users/brad9/Desktop/BirdCLEF/log_mel_32000kHZ"
CONFIG_PATH = "C:/Users/brad9/Desktop/BirdCLEF/audio_preprocessing_config.json"
BATCH_SIZE = 64
EPOCHS = 30
FIXED_MEL_LENGTH = 313
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH) as f:
    config = json.load(f)
with open(TO_Organism) as f:
    idx_to_organism = json.load(f)
with open(TO_IDX) as f:
    organism_class_to_idx = json.load(f)

# ==== 數據集：支援所有config內數據增強（訓練用）====
class LogMelOrganismDataset(Dataset):
    def __init__(self, df, mel_dir, mask, class_map, config, augment=True):
        self.df = df
        self.mel_dir = mel_dir
        self.mask = mask
        self.class_map = class_map
        self.config = config
        self.augment = augment
        self.id_map = {self.class_map[label]: i for i, label in enumerate(mask)}
        # 確保 random_time_shift 有在 config
        if "random_time_shift" not in self.config["augmentation_strategy"]:
            self.config["augmentation_strategy"]["random_time_shift"] = {"apply": True, "max_shift_ratio": 0.1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_path = os.path.join(self.mel_dir, row["mel_filename"].replace("\\", "/"))
        mel = np.load(mel_path)
        if mel.shape[0] != 128 and mel.shape[1] == 128:
            mel = mel.T
        assert mel.shape[0] == 128, f"Mel shape[0] must be 128, got {mel.shape}"
        mel = (mel - mel.mean()) / (mel.std() + self.config["normalization"].get("epsilon", 1e-6))

        # ====== 數據增強 ======
        if self.augment:
            # SpecAugment
            specaug = self.config["augmentation_strategy"].get("specaugment", {})
            if specaug.get("apply", False):
                if random.random() < 0.5:
                    fmask = specaug.get("frequency_mask_param", 12)
                    f0 = random.randint(0, 128 - fmask)
                    mel[f0:f0+fmask, :] = 0
                if random.random() < 0.5:
                    T = mel.shape[1]
                    tmask = specaug.get("time_mask_param", 24)
                    t0 = random.randint(0, max(1, T - tmask))
                    mel[:, t0:t0+tmask] = 0
            # Random time shift
            shift_conf = self.config["augmentation_strategy"].get("random_time_shift", {})
            if shift_conf.get("apply", True):
                max_shift = int(mel.shape[1] * shift_conf.get("max_shift_ratio", 0.1))
                if max_shift > 0 and random.random() < 0.5:
                    shift = random.randint(0, max_shift)
                    mel = np.roll(mel, shift, axis=1)
            # 加性雜訊
            if self.config["augmentation"].get("add_noise", False) or \
                self.config["augmentation_strategy"].get("additive_noise", {}).get("apply", False):
                snr_db = self.config["augmentation"].get("noise_snr_db",
                    self.config["augmentation_strategy"].get("additive_noise", {}).get("snr_db", 20))
                noise = np.random.randn(*mel.shape)
                rms_signal = np.sqrt(np.mean(mel ** 2))
                rms_noise = np.sqrt(np.mean(noise ** 2))
                desired_rms_noise = rms_signal / (10 ** (snr_db / 20))
                mel += noise * (desired_rms_noise / (rms_noise + 1e-9))
            # 隨機裁切
            crop_conf = self.config["augmentation_strategy"].get("random_time_crop", {"apply": False, "crop_ratio": 0.9})
            if crop_conf.get("apply", False) and random.random() < 0.5:
                crop_ratio = crop_conf.get("crop_ratio", 0.9)
                crop_len = int(mel.shape[1] * crop_ratio)
                if crop_len < mel.shape[1]:
                    start = random.randint(0, mel.shape[1] - crop_len)
                    mel = mel[:, start:start+crop_len]
                    pad = FIXED_MEL_LENGTH - crop_len
                    mel = np.pad(mel, ((0,0),(0, pad)), mode='constant')

        mel = torch.from_numpy(mel).float().unsqueeze(0)  # [1, 128, T]

        # 處理 multi-label one-hot 標籤
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


        
        return mel, label

# ==== EfficientNetB3 主模型 ====
class MelEfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, in_chans=1)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.backbone(x)

# ==== DataLoader 工具 ====
def get_loaders(organism_type):
    assert organism_type in MASK_PATHS, f"不支援的類別: {organism_type}"
    with open(MASK_PATHS[organism_type]) as f:
        class_mask_idx = json.load(f)
    class_mask = [idx_to_organism[str(i)] for i in class_mask_idx]
    df = pd.read_csv(CSV_PATH)
    df = df[df["primary_label"].isin(class_mask)]
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = LogMelOrganismDataset(train_df, MEL_DIR, class_mask, organism_class_to_idx, config, augment=True)
    val_ds = LogMelOrganismDataset(val_df, MEL_DIR, class_mask, organism_class_to_idx, config, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, class_mask

# ==== 主訓練程式 ====
def train_efficientnet_b3(
    organism_type,
    patience=6,
    tta_times=5,
    amp_enable=True,
    warmup_epochs=3
):
    train_loader, val_loader, class_mask = get_loaders(organism_type)
    model = MelEfficientNetB3(num_classes=len(class_mask)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=amp_enable)
    # Warmup + ReduceLROnPlateau
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_val_map = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=amp_enable):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_train_loss = total_loss / len(train_loader)
        print(f"[{organism_type}] Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_scheduler.step()

        # ====== 驗證 (含 TTA) ======
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds_tta = []
                for _ in range(tta_times):
                    x_tta = x.clone()
                    for i in range(x_tta.shape[0]):
                        mel_np = x_tta[i][0].cpu().numpy()
                        # SpecAugment
                        specaug = config["augmentation_strategy"].get("specaugment", {})
                        if specaug.get("apply", False):
                            if random.random() < 0.5:
                                fmask = specaug.get("frequency_mask_param", 12)
                                f0 = random.randint(0, 128 - fmask)
                                mel_np[f0:f0+fmask, :] = 0
                            if random.random() < 0.5:
                                T = mel_np.shape[1]
                                tmask = specaug.get("time_mask_param", 24)
                                t0 = random.randint(0, max(1, T - tmask))
                                mel_np[:, t0:t0+tmask] = 0
                        # random time shift
                        shift_conf = config["augmentation_strategy"].get("random_time_shift", {})
                        if shift_conf.get("apply", True):
                            max_shift = int(mel_np.shape[1] * shift_conf.get("max_shift_ratio", 0.1))
                            if max_shift > 0 and random.random() < 0.5:
                                shift = random.randint(0, max_shift)
                                mel_np = np.roll(mel_np, shift, axis=1)
                        x_tta[i][0] = torch.from_numpy(mel_np)
                    with autocast(enabled=amp_enable):
                        logits = model(x_tta)
                        preds_tta.append(torch.sigmoid(logits).cpu())
                preds = torch.stack(preds_tta).mean(dim=0)
                val_loss += criterion(logits, y).item()
                all_preds.append(preds)
                all_labels.append(y.cpu())
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        # mAP
        try:
            val_map = average_precision_score(labels, preds, average="macro")
        except Exception:
            val_map = 0
        avg_val_loss = val_loss / len(val_loader)
        print(f"[{organism_type}] Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, mAP: {val_map:.4f}")

        # ReduceLROnPlateau 根據 mAP 調整
        plateau_scheduler.step(val_map)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # EarlyStopping based on mAP
        if val_map > best_val_map:
            best_val_map = val_map
            patience_counter = 0
            model_path = f"best_{organism_type}_efficientnetb3.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  [Checkpoint] New best model saved as {model_path}. mAP: {best_val_map:.4f}")
        else:
            patience_counter += 1
            print(f"  [EarlyStopping] No improvement for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f"  [EarlyStopping] Stop at epoch {epoch+1}. Best mAP: {best_val_map:.4f}")
                break

if __name__ == "__main__":
    train_efficientnet_b3("Aves", patience=6, tta_times=5, amp_enable=True, warmup_epochs=3)
