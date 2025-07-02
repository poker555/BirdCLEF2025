import os
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score
import random
import pandas as pd

# ==== 基本參數 ====
HDF5_PATH = "C:/Users/brad9/Desktop/BirdCLEF/all_mel_data.h5"
RECORD_PATH = "C:/Users/brad9/Desktop/BirdCLEF/hdf5_index_record.csv"
TAXONOMY_PATH = "C:/Users/brad9/Desktop/BirdCLEF/taxonomy.csv"
CONFIG_PATH = "C:/Users/brad9/Desktop/BirdCLEF/audio_preprocessing_config.json"
BATCH_SIZE = 64
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_FILES = {
    "Amphibia": "C:/Users/brad9/Desktop/BirdCLEF/index/organism_masks/class_mask_species_Amphibia.json",
    "Aves": "C:/Users/brad9/Desktop/BirdCLEF/index/organism_masks/class_mask_species_Aves.json",
    "Insecta": "C:/Users/brad9/Desktop/BirdCLEF/index/organism_masks/class_mask_species_Insecta.json",
    "Mammalia": "C:/Users/brad9/Desktop/BirdCLEF/index/organism_masks/class_mask_species_Mammalia.json",
}

with open(CONFIG_PATH) as f:
    config = json.load(f)

# ==== taxonomy.csv 建立 primary_label 對 species_class 的 mapping ====
taxonomy_df = pd.read_csv(TAXONOMY_PATH)
# 根據你的 taxonomy 欄位，如果名稱不是 primary_label/species_class 請修改下方這兩個
primarylabel_to_species = dict(zip(
    taxonomy_df["primary_label"].astype(str),
    taxonomy_df["class_name"]
))

# ==== Dataset 和主程式不變 ====
class HDF5MelDataset(Dataset):
    def __init__(self, hdf5_path, indices=None, class_mask=None, augment=False, config=None):
        self.hdf5_path = hdf5_path
        self.augment = augment
        self.config = config
        self.class_mask = class_mask
        self.h5f = h5py.File(hdf5_path, 'r')
        self.mel_ds = self.h5f['mel']
        self.label_ds = self.h5f['label']
        self.indices = np.arange(len(self.mel_ds)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        mel = self.mel_ds[i]
        label = self.label_ds[i]
        if self.class_mask is not None:
            label = label[self.class_mask]
        if self.augment and self.config is not None:
            mel = self.apply_augmentation(mel)
        return torch.from_numpy(mel).float(), torch.from_numpy(label).float()

    def apply_augmentation(self, mel):
        mel = mel[0]
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
        shift_conf = self.config["augmentation_strategy"].get("random_time_shift", {})
        if shift_conf.get("apply", True):
            max_shift = int(mel.shape[1] * shift_conf.get("max_shift_ratio", 0.1))
            if max_shift > 0 and random.random() < 0.5:
                shift = random.randint(0, max_shift)
                mel = np.roll(mel, shift, axis=1)
        return mel[np.newaxis, :, :]

    def __del__(self):
        try:
            self.h5f.close()
        except:
            pass

class MelEfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, in_chans=1)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.backbone(x)

# ==== 重點：taxonomy檔案進行正確比對 ====
def get_species_indices_and_mask(record_path, organism_name):
    record = pd.read_csv(record_path)
    record["primary_label"] = record["mel_filename"].apply(lambda x: str(x.split("/")[0]))
    organism_indices = []
    for idx, row in record.iterrows():
        primary_label = row["primary_label"]
        org_class = primarylabel_to_species.get(primary_label, None)
        if org_class == organism_name:
            organism_indices.append(row["index"])
    with open(MASK_FILES[organism_name]) as f:
        class_mask = np.array(json.load(f))
    return np.array(organism_indices, dtype=int), class_mask

def get_hdf5_loaders_for_species(
    hdf5_path, record_path, organism_name, config, val_ratio=0.2, batch_size=64
):
    indices, class_mask = get_species_indices_and_mask(record_path, organism_name)
    if len(indices) == 0:
        raise ValueError(f"[Error] No samples found for organism '{organism_name}'. 請確認 taxonomy.csv 與 mapping！")
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42)
    train_ds = HDF5MelDataset(hdf5_path, train_idx, class_mask=class_mask, augment=True, config=config)
    val_ds = HDF5MelDataset(hdf5_path, val_idx, class_mask=class_mask, augment=False, config=config)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, len(class_mask), class_mask

# ==== 主訓練程式（不變） ====
def train_efficientnetb3_hdf5_species(
    organism_name,
    patience=6,
    tta_times=5,
    amp_enable=True,
    warmup_epochs=3,
    batch_size=64
):
    train_loader, val_loader, num_classes, class_mask = get_hdf5_loaders_for_species(
        HDF5_PATH, RECORD_PATH, organism_name, config, batch_size=batch_size)
    model = MelEfficientNetB3(num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=amp_enable)
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
        print(f"[{organism_name}] Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

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
        try:
            val_map = average_precision_score(labels, preds, average="macro")
        except Exception:
            val_map = 0
        avg_val_loss = val_loss / len(val_loader)
        print(f"[{organism_name}] Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, mAP: {val_map:.4f}")

        plateau_scheduler.step(val_map)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_map > best_val_map:
            best_val_map = val_map
            patience_counter = 0
            model_path = f"best_{organism_name}_efficientnetb3_hdf5.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  [Checkpoint] New best model saved as {model_path}. mAP: {best_val_map:.4f}")
        else:
            patience_counter += 1
            print(f"  [EarlyStopping] No improvement for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f"  [EarlyStopping] Stop at epoch {epoch+1}. Best mAP: {best_val_map:.4f}")
                break

if __name__ == "__main__":
    train_efficientnetb3_hdf5_species(
        organism_name="Aves", patience=6, tta_times=5, amp_enable=True, warmup_epochs=3, batch_size=BATCH_SIZE
    )
