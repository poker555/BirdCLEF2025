import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import timm
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import os
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Dataset Definition --------
class BirdclefH5Dataset(Dataset):
    def __init__(self, h5_path, split='train', augment=False):
        self.h5_path = h5_path
        self.augment = augment
        self.split = split.encode()

        with h5py.File(h5_path, 'r') as h5f:
            self.indices = [i for i, s in enumerate(h5f['split']) if s == self.split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        with h5py.File(self.h5_path, 'r') as h5:
            x = h5['spectrogram'][i].astype(np.float32)
            y = h5['multi_label'][i].astype(np.float32)

        if self.augment:
            x = self.apply_augment(x)

        x = np.expand_dims(x, axis=0)  # shape: (1, 128, 313)
        return torch.tensor(x), torch.tensor(y)

    def apply_augment(self, x):
        if random.random() < 0.5:
            t = random.randint(0, x.shape[1] - 1)
            t_width = random.randint(10, 30)
            x[:, t:t+t_width] = 0
        if random.random() < 0.5:
            f = random.randint(0, x.shape[0] - 1)
            f_width = random.randint(5, 15)
            x[f:f+f_width, :] = 0
        if random.random() < 0.5:
            shift = random.randint(-20, 20)
            x = np.roll(x, shift, axis=1)
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            x = np.roll(x, shift, axis=0)
        return x

# -------- Model --------
def get_model(num_classes):
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes,in_chans=1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model

# -------- EarlyStopping --------
class EarlyStopping:
    def __init__(self, patience=10, mode='max', min_delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.early_stop = False
        self.mode = mode
        self.min_delta = min_delta

    def __call__(self, current_score):
        improved = (current_score - self.best > self.min_delta) if self.mode == 'max' else (self.best - current_score > self.min_delta)
        if improved:
            self.best = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# -------- Training Loop --------
def train_one_epoch(model, loader, optimizer, scaler, loss_fn):
    model.train()
    total_loss = 0
    for X, y in tqdm(loader, desc="Train"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            logits = model(X)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Val"):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    return total_loss / len(loader), all_preds, all_targets

# -------- Main Training Entry --------
def train():
    h5_path = r"C:\Users\brad9\Desktop\BirdCLEF\birdclef_data.h5"
    h5 = h5py.File(h5_path, 'r')
    num_classes = h5['multi_label'].shape[1]

    # Compute per-sample weights for oversampling
    train_idx = [i for i, s in enumerate(h5['split']) if s == b'train']
    sample_weights = [h5['weight'][i] for i in train_idx]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_idx), replacement=True)

    train_set = BirdclefH5Dataset(h5_path, 'train', augment=True)
    val_set = BirdclefH5Dataset(h5_path, 'val', augment=False)

    train_loader = DataLoader(train_set, batch_size=16, sampler=sampler, num_workers=10, pin_memory=True,
    persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=10, pin_memory=True,
    persistent_workers=True)

    model = get_model(num_classes).to(device)
    loss_fn = nn.BCEWithLogitsLoss()  # multi-label binary loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()

    early_stopper = EarlyStopping(patience=10, mode='max', min_delta=1e-4)
    best_macro = -1.0

    for epoch in range(100):
        print(f"\nEpoch {epoch+1}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn)
        val_loss, y_pred, y_true = validate(model, val_loader, loss_fn)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        y_pred_bin = (y_pred > 0.5).astype(int)
        f1 = f1_score(y_true, y_pred_bin, average='macro')
        print(f"Val Macro F1: {f1:.4f}")

        # 儲存模型條件改為 macro
        if f1 - early_stopper.min_delta > best_macro:
            best_macro = f1
            torch.save(model.state_dict(), "best_model.pt")
            print("Model saved.")

        early_stopper(f1)  # 改為監控 macro
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Output classification report
    report = classification_report(y_true, y_pred_bin, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("classification_report.csv")
    print("Classification report saved.")

if __name__ == "__main__":
    train()
