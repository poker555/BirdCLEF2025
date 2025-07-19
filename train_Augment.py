import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm
import numpy as np
import random

# ---- 1. HDF5 Dataset ----
class H5MelDataset(Dataset):
    def __init__(self, h5_path, split='train', img_size=300):
        self.h5_path = h5_path
        self.img_size = img_size
        self.h5 = None
        with h5py.File(h5_path, 'r') as h5:
            all_splits = h5['split'][:]
            all_splits = [s.decode() if isinstance(s, bytes) else s for s in all_splits]
            self.indices = [i for i, s in enumerate(all_splits) if s == split]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, 'r')
        real_idx = self.indices[idx]
        mel = self.h5['mel'][real_idx]
        label = self.h5['label'][real_idx]
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0).repeat(3, 1, 1)
        mel = torch.nn.functional.interpolate(
            mel.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False
        ).squeeze(0)
        label = torch.tensor(label, dtype=torch.float32)
        return mel, label

# ---- 2. EfficientNet-B3 ----
def get_model(num_classes):
    model = timm.create_model('efficientnet_b3', pretrained=True, in_chans=3)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# ---- 3. EarlyStopping ----
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

# ---- 4. Mixup (for batch) ----
def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

# ---- 5. Label Smoothing for Multi-label ----
def apply_label_smoothing(labels, smoothing=0.02):
    return labels * (1 - smoothing) + smoothing * (1 - labels)

# ---- 6. Focal Loss for Multi-label ----
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

# ---- 7. 訓練主程式 ----
def main():
    # ==== 檢查 GPU 狀態 ====
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available. Only CPU will be used.")

    # ==== 可調整參數區 ====
    BATCH_SIZE = 32
    H5_PATH = r"C:\Users\brad9\Desktop\BirdCLEF++\train_withAugment.h5"
    NUM_EPOCHS = 100
    MIXUP = True            # <<<<--- 是否啟用 Mixup
    LABEL_SMOOTH = True     # <<<<--- 是否啟用 Label Smoothing
    FOCAL_LOSS = True      # <<<<--- 是否啟用 Focal Loss (True則用Focal, False則用BCE)
    LR = 2e-4
    SMOOTHING_VAL = 0.02    # Label Smoothing 強度
    MIXUP_ALPHA = 0.4
    # ======================

    train_set = H5MelDataset(H5_PATH, split='train', img_size=300)
    valid_set = H5MelDataset(H5_PATH, split='valid', img_size=300)
    with h5py.File(H5_PATH, 'r') as h5f:
        num_classes = h5f['label'].shape[1]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(num_classes).to(device)
    if FOCAL_LOSS:
        criterion = BCEFocalLoss(gamma=2, alpha=0.25)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    early_stopper = EarlyStopping(patience=7, verbose=True, save_path="best_model_v2.pth")
    scaler = torch.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, n = 0, 0
        for mels, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            mels = mels.to(device)
            labels = labels.to(device)
            # -- Mixup --
            if MIXUP:
                mels, labels = mixup_data(mels, labels, alpha=MIXUP_ALPHA)
            # -- Label Smoothing --
            if LABEL_SMOOTH:
                labels = apply_label_smoothing(labels, smoothing=SMOOTHING_VAL)
            optimizer.zero_grad()
            device_type = 'cuda'
            with torch.amp.autocast(device_type):
                logits = model(mels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * mels.size(0)
            n += mels.size(0)
        print(f"[Train] Epoch {epoch+1} loss: {train_loss/n:.4f}")

        # 驗證（不做Mixup/LabelSmooth）
        model.eval()
        val_loss, n = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mels, labels in tqdm(valid_loader, desc=f"Valid Epoch {epoch+1}"):
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

        # 動態學習率調整
        scheduler.step(macro_f1)
        # Early Stopping & Save
        early_stopper(macro_f1, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print("訓練結束！最佳模型已存為 best_model_v3.pth")

if __name__ == "__main__":
    main()
