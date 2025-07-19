import os
import h5py
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm
import albumentations as A
import albumentations.pytorch

# ======= 設定讀取 =======
with open(r'C:\Users\brad9\Desktop\BirdCLEF\config_train.json') as f:
    config = json.load(f)

train_cfg = config['training']
aug_cfg_all = config['augmentation_strategy']

# ======= Dataset =======
class BirdH5Dataset(Dataset):
    def __init__(self, h5_path, split='train', augment=True):
        self.h5_path = h5_path
        self.split = split
        self.augment = augment

        with h5py.File(h5_path, 'r') as h5f:
            self.indices = [i for i in range(len(h5f['mels'])) if h5f['split'][i].decode() == split]
            self.num_classes = h5f['onehot'].shape[1]

    def __len__(self):
        return len(self.indices)

    def get_augment(self, quality):
        if quality >= 4:
            cfg = aug_cfg_all['quality_oriented']['high']
        elif quality >= 2:
            cfg = aug_cfg_all['quality_oriented']['mid']
        else:
            cfg = aug_cfg_all['quality_oriented']['low']

        transforms = []

        if cfg.get('specaugment', {}).get('apply', False):
            p = cfg['specaugment'].get('p', 0.15)
            transforms.append(A.FrequencyMasking(freq_mask_param=cfg['specaugment']['frequency_mask_param'], p=p))
            transforms.append(A.TimeMasking(time_mask_param=cfg['specaugment']['time_mask_param'], p=p))

        if cfg.get('gaussian_blur', {}).get('apply', False):
            transforms.append(A.GaussianBlur(
                blur_limit=cfg['gaussian_blur']['kernel_size'],
                p=cfg['gaussian_blur']['p']
            ))

        if cfg.get('random_erasing', {}).get('apply', False):
            transforms.append(A.CoarseDropout(
                min_height=int(cfg['random_erasing']['rect_ratio'][0] * 128),
                max_height=int(cfg['random_erasing']['rect_ratio'][1] * 128),
                min_width=int(cfg['random_erasing']['rect_ratio'][0] * 313),
                max_width=int(cfg['random_erasing']['rect_ratio'][1] * 313),
                fill_value=0,
                p=cfg['random_erasing']['p']
            ))

        if cfg.get('contrast', {}).get('apply', False):
            transforms.append(A.RandomBrightnessContrast(
                contrast_limit=(cfg['contrast']['contrast_range'][0] - 1, cfg['contrast']['contrast_range'][1] - 1),
                brightness_limit=0,
                p=cfg['contrast']['p']
            ))

        transforms.append(A.Normalize(mean=0, std=1))
        transforms.append(A.pytorch.ToTensorV2())
        return A.Compose(transforms)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5:
            i = self.indices[idx]
            mel = h5['mels'][i]
            label = h5['onehot'][i]
            weight = h5['weight'][i]
            quality = h5['quality'][i]

            if self.augment and self.split == 'train':
                aug = self.get_augment(quality)
                mel = aug(image=mel)['image']
            else:
                mel = torch.tensor(mel).float()

            label = torch.tensor(label).float()
            weight = torch.tensor(weight).float()

            strategy = aug_cfg_all['quality_oriented']
            if quality >= 4:
                mix_cfg = strategy['high'].get('mixup', {})
            elif quality >= 2:
                mix_cfg = strategy['mid'].get('mixup', {})
            else:
                mix_cfg = strategy['low'].get('mixup', {})

            if self.split == 'train' and mix_cfg.get("apply", False) and random.random() < mix_cfg.get("p", 0.2):
                j = random.choice(self.indices)
                mel2 = h5['mels'][j]
                label2 = h5['onehot'][j]
                weight2 = h5['weight'][j]
                quality2 = h5['quality'][j]

                if self.augment:
                    aug2 = self.get_augment(quality2)
                    mel2 = aug2(image=mel2)['image']
                else:
                    mel2 = torch.tensor(mel2).unsqueeze(0).float()

                label2 = torch.tensor(label2).float()
                weight2 = torch.tensor(weight2).float()

                lam = np.random.beta(mix_cfg.get("alpha", 0.2), mix_cfg.get("alpha", 0.2))
                mel = lam * mel + (1 - lam) * mel2
                label = lam * label + (1 - lam) * label2
                weight = lam * weight + (1 - lam) * weight2

        return mel, label, weight


# ======= Model =======
class BirdEffNetB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, in_chans=1)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

# ======= 評估 =======
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0

    with torch.no_grad():
        for mel, label, weight in tqdm(loader, desc="Valid"):
            mel, label, weight = mel.to(device), label.to(device), weight.to(device)
            out = model(mel)
            loss = criterion(out, label)
            loss = (loss.mean(dim=1) * weight).mean()
            val_loss += loss.item()

            all_preds.append(torch.sigmoid(out).detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = (np.vstack(all_preds) > 0.5).astype(np.int32)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return val_loss / len(loader), macro

# ======= 主訓練 =======
def train(h5_path='train_valid_weighted.h5'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = BirdH5Dataset(h5_path, split='train', augment=True)
    val_ds = BirdH5Dataset(h5_path, split='valid', augment=False)

    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True,pin_memory=True,
    persistent_workers=True, num_workers=10)
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'], shuffle=False,pin_memory=True,
    persistent_workers=True, num_workers=10)

    model = BirdEffNetB3(num_classes=train_ds.num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    opt_cfg = train_cfg['optimizer'].copy()
    opt_type = opt_cfg.pop('type')
    if opt_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **opt_cfg)
    elif opt_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **opt_cfg)
    else:
        raise ValueError(f"[錯誤] 未支援 optimizer type: {opt_type}")

    sch_cfg = train_cfg['scheduler'].copy()
    sch_type = sch_cfg.pop('type')
    if sch_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sch_cfg)
    elif sch_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sch_cfg)
    else:
        raise ValueError(f"[錯誤] 未支援 scheduler type: {sch_type}")

    best_macro = -1.0
    patience = 0
    scaler = torch.amp.GradScaler(device='cuda', enabled=train_cfg.get("mixed_precision", True))

    for epoch in range(train_cfg['epochs']):
        model.train()
        total_loss = 0
        for mel, label, weight in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            mel, label, weight = mel.to(device), label.to(device), weight.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=train_cfg.get("mixed_precision", True)):
                out = model(mel)
                loss = criterion(out, label)
                loss = (loss.mean(dim=1) * weight).mean()

            scaler.scale(loss).backward()
            if train_cfg.get("gradient_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        val_loss, val_macro = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro:.4f}")

        if val_macro - train_cfg['early_stopping']['min_delta'] > best_macro:
            best_macro = val_macro
            torch.save(model.state_dict(), "best_model.pth")
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg['early_stopping']['patience']:
                print("✅ Early stopping triggered.")
                break

if __name__ == "__main__":
    train(h5_path=r"C:\\Users\\brad9\\Desktop\\BirdCLEF\\train_valid_weighted.h5")
