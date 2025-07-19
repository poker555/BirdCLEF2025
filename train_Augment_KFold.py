import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import os
import json
import torchaudio

# ========== 增強函數 ==========
def apply_specaugment(mels, cfg):
    """mels: [B, 3, H, W]"""
    if not cfg.get("apply", False):
        return mels
    p = cfg.get("p", 0.5)
    freq_mask_param = cfg.get("frequency_mask_param", 12)
    time_mask_param = cfg.get("time_mask_param", 24)
    for i in range(mels.size(0)):
        if np.random.rand() < p:
            mels[i] = torchaudio.transforms.FrequencyMasking(freq_mask_param)(mels[i])
            mels[i] = torchaudio.transforms.TimeMasking(time_mask_param)(mels[i])
    return mels

def apply_mixup(mels, labels, cfg):
    if not cfg.get("apply", False) or np.random.rand() > cfg.get("p", 0.2):
        return mels, labels
    alpha = cfg.get("alpha", 0.2)
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(mels.size(0))
    mixed_mels = lam * mels + (1 - lam) * mels[index, :]
    mixed_labels = lam * labels + (1 - lam) * labels[index, :]
    return mixed_mels, mixed_labels

# ========== 資料集 ==========
class H5MelDataset(Dataset):
    def __init__(self, h5_path, meta, fold, split='train', img_size=300):
        self.h5_path = h5_path
        self.img_size = img_size
        self.h5 = None
        if split == 'train':
            self.meta = meta[meta['fold'] != fold].reset_index(drop=True)
        else:
            self.meta = meta[meta['fold'] == fold].reset_index(drop=True)
        self.indices = self.meta.index.values
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, 'r')
        real_idx = self.indices[idx]
        mel = self.h5['mel'][real_idx]
        label = self.h5['label'][real_idx]
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0).repeat(3, 1, 1)  # 灰階複製為3ch
        mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
        label = torch.tensor(label, dtype=torch.float32)
        return mel, label, real_idx

# ========== 模型與損失 ==========
def get_model(num_classes):
    model = timm.create_model('efficientnet_b3', pretrained=True, in_chans=3)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

class EarlyStopping:
    def __init__(self, patience=4, verbose=False, save_path="best_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Best model saved (macro F1: {score:.4f})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1-pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

# ========== KFold主流程 ==========
def train_kfold(meta_csv, h5_path, config_path, num_folds=3, batch_size=32, num_epochs=100, img_size=300, patience=7, lr=2e-4):
    meta = pd.read_csv(meta_csv)
    with h5py.File(h5_path, 'r') as h5f:
        num_classes = h5f['label'].shape[1]
    # 載入增強策略
    with open(config_path) as f:
        cfg_json = json.load(f)
    augment_cfg = cfg_json["augmentation_strategy"]["neutral"]

    fold_macro, fold_micro = [], []
    for fold in range(num_folds):
        print(f"\n=== Fold {fold+1}/{num_folds} ===")
        train_set = H5MelDataset(h5_path, meta, fold, split='train', img_size=img_size)
        valid_set = H5MelDataset(h5_path, meta, fold, split='valid', img_size=img_size)

        if 'sample_weight' in meta.columns:
            train_weights = meta.iloc[train_set.indices]['sample_weight'].values
            train_sampler = WeightedRandomSampler(train_weights, len(train_set), replacement=True)
            train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = get_model(num_classes).to(device)
        criterion = BCEFocalLoss(gamma=2, alpha=0.25)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        early_stopper = EarlyStopping(patience=patience, verbose=True, save_path=f"best_model_fold{fold}.pth")
        scaler = torch.amp.GradScaler()

        for epoch in range(num_epochs):
            model.train()
            train_loss, n = 0, 0
            for mels, labels, _ in tqdm(train_loader, desc=f"Fold {fold+1} Train Epoch {epoch+1}"):
                mels = mels.to(device)
                labels = labels.to(device)
                # --- SpecAugment ---
                if 'specaugment' in augment_cfg:
                    mels = apply_specaugment(mels, augment_cfg['specaugment'])
                # --- Mixup ---
                if 'mixup' in augment_cfg:
                    mels, labels = apply_mixup(mels, labels, augment_cfg['mixup'])

                optimizer.zero_grad()
                device = 'cuda'
                with torch.amp.autocast(device_type=device):
                    logits = model(mels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * mels.size(0)
                n += mels.size(0)
            print(f"[Train] Epoch {epoch+1} loss: {train_loss/n:.4f}")

            model.eval()
            val_loss, n = 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for mels, labels, _ in tqdm(valid_loader, desc=f"Fold {fold+1} Valid Epoch {epoch+1}"):
                    mels = mels.to(device)
                    labels = labels.to(device)
                    logits = model(mels)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * mels.size(0)
                    n += mels.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.append(probs)
                    all_labels.append(labels.cpu().numpy())
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            macro_f1 = f1_score(all_labels, all_preds > 0.5, average='macro', zero_division=0)
            micro_f1 = f1_score(all_labels, all_preds > 0.5, average='micro', zero_division=0)
            print(f"[Valid] Epoch {epoch+1} loss: {val_loss/n:.4f} | macro F1: {macro_f1:.4f} | micro F1: {micro_f1:.4f}")

            scheduler.step(macro_f1)
            early_stopper(macro_f1, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

        print(f"Fold {fold+1} 訓練結束，最佳模型已存為 best_model_fold{fold}.pth")
        fold_macro.append(macro_f1)
        fold_micro.append(micro_f1)
    print(f"\n所有 Fold 訓練完畢，平均 macro F1 = {np.mean(fold_macro):.4f}, micro F1 = {np.mean(fold_micro):.4f}")

if __name__ == "__main__":
    train_kfold(
        meta_csv=r"C:\Users\brad9\Desktop\BirdCLEF\code\meta_all.csv",
        h5_path=r"C:\Users\brad9\Desktop\BirdCLEF\train_all_withAugment.h5",
        config_path=r"C:\Users\brad9\Desktop\BirdCLEF\config.json",
        num_folds=3,
        batch_size=32,
        num_epochs=50,
        img_size=300,
        patience=7,
        lr=2e-4
    )
