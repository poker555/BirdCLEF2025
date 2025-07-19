import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm
import numpy as np

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

    # ---- 4. DataLoader/準備 ----
    BATCH_SIZE = 32
    H5_PATH = r"C:\Users\brad9\Desktop\BirdCLEF++\train_v2.h5"
    train_set = H5MelDataset(H5_PATH, split='train', img_size=300)
    valid_set = H5MelDataset(H5_PATH, split='valid', img_size=300)
    with h5py.File(H5_PATH, 'r') as h5f:
        num_classes = h5f['label'].shape[1]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    # ---- 5. 訓練主程式 (AMP) ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
    early_stopper = EarlyStopping(patience=4, verbose=True, save_path="best_model_v2.pth")
    scaler = torch.cuda.amp.GradScaler()  # 混合精度
    NUM_EPOCHS = 100

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, n = 0, 0
        for mels, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            mels = mels.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(mels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * mels.size(0)
            n += mels.size(0)
        print(f"[Train] Epoch {epoch+1} loss: {train_loss/n:.4f}")

        # 驗證（不需要AMP）
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

    print("訓練結束！最佳模型已存為 best_model.pth")

if __name__ == "__main__":
    main()
