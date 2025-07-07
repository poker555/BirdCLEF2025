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
from torch.amp import GradScaler ,autocast
from sklearn.metrics import average_precision_score
import random
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchaudio
from datetime import datetime

# ==== 基本參數 ====
HDF5_PATH = "C:/Users/brad9/Desktop/BirdCLEF/log_mel_data.h5"
RECORD_PATH = "C:/Users/brad9/Desktop/BirdCLEF/log_mel.csv"
Train_PATH = "C:/Users/brad9/Desktop/BirdCLEF/merge_train.csv"
CONFIG_PATH = "C:/Users/brad9/Desktop/BirdCLEF/audio_preprocessing_config.json"

BATCH_SIZE = 64
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Label = 3
name = "Aves"  # 這裡可以根據需要修改為其他類別


with open(CONFIG_PATH) as f:
        config = json.load(f)

    # ==== taxonomy.csv 建立 primary_label 對 species_class 的 mapping ====
train_df = pd.read_csv(Train_PATH)
 
    #時間平移
def random_time_shift(mel,shift_ratio):
        T = mel.shape[1]
        shift = int(T * shift_ratio)
        if shift <1:
            return mel
        shift = random.randint(-shift, shift)
        shifted_mel = np.roll(mel, shift, axis=1)
        return shifted_mel


                           
#數據增強Pipeline
def Augment(aug_config):
        voice = aug_config["augmentation_strategy"]["additive_noise"]["snr_db"]
        pipeline = []
        if aug_config["augmentation_strategy"]["additive_noise"]["apply"]:
            pipeline.append(A.GaussNoise(
                
                var_limit=(voice,voice),
                p = aug_config["augmentation_strategy"]["additive_noise"].get("p", 1.0),
            ))
        # ...其他 albumentations 支援的增強...
        pipeline.append(ToTensorV2(transpose_mask=False))
        return A.Compose(pipeline)
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, val_score):
        score = val_score
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
class HDF5Mel_Dataset(Dataset):
        def __init__(self,hdf5_path,idx_list,label):
            self.hdf5_path = hdf5_path
            self.config = config
            self.target = label
            self.idx_list = list(idx_list)
            self.pipeline = Augment(self.config)
            self.h5_file = None
            
        
        def __len__(self):
            return len(self.idx_list)
        
        def __getitem__(self, idx):
        # 1. 直接從 h5 取出 numpy array (128, T)
            if self.h5_file is None:
                self.h5_file = h5py.File(self.hdf5_path, 'r')
                self.mel_data = self.h5_file['mel']
                self.class_labels = self.h5_file['class_label']
                self.class_names = {
                    'Insecta': self.h5_file['class_name_Insecta'],
                    'Amphibia': self.h5_file['class_name_Amphibia'],
                    'Mammalia': self.h5_file['class_name_Mammalia'],
                    'Aves': self.h5_file['class_name_Aves']
                }
                
            real_idx = self.idx_list[idx]
            mel = self.mel_data[real_idx][0]
            
            # 2. SpecAugment (torchaudio，需先轉 torch tensor shape (1, 128, T))
            if self.config["augmentation_strategy"]["specaugment"]["apply"]:
                mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
                freq_mask = torchaudio.transforms.FrequencyMasking(
                    freq_mask_param=self.config["augmentation_strategy"]["specaugment"]["frequency_mask_param"]
                )
                time_mask = torchaudio.transforms.TimeMasking(
                    time_mask_param=self.config["augmentation_strategy"]["specaugment"]["time_mask_param"]
                )
                mel_tensor = freq_mask(mel_tensor)
                mel_tensor = time_mask(mel_tensor)
                mel = mel_tensor.squeeze(0).numpy()   # (128, T)
            
            # 3. random_time_shift (numpy)
            if self.config["augmentation_strategy"]["random_time_shift"]["apply"]:
                max_shift_ratio = self.config["augmentation_strategy"]["random_time_shift"]["max_shift_ratio"]
                mel = random_time_shift(mel, max_shift_ratio)
            
            # 4. albumentations pipeline (需要 shape (128, T, 1))
            mel = np.expand_dims(mel, axis=-1)
            if self.pipeline:
                mel = self.pipeline(image=mel)["image"]    # torch tensor, (1, 128, T)
            else:
                mel = torch.tensor(mel.transpose(2,0,1), dtype=torch.float32) # (1, 128, T)
            


            match self.target:
                case 0:
                    class_label = self.class_names['Insecta'][real_idx]
                case 1:
                    class_label = self.class_names['Amphibia'][real_idx]
                case 2:
                    class_label = self.class_names['Mammalia'][real_idx]
                case 3:
                    class_label = self.class_names['Aves'][real_idx]
                case _:
                    raise ValueError(f"Unknown label: {self.target}")
                
            if isinstance(class_label, bytes):
                class_label = class_label.decode('utf-8')
            class_label = int(float(class_label))
            
            return {
                'mel': mel,
                'class_label': class_label,
            }
            
def main():




    record_df = pd.read_csv(RECORD_PATH)
    record_df['audio_id'] = record_df['mel_filename'].apply(lambda x: '\\'.join(x.split('\\')[:2]))
    filtered_df = record_df[record_df['class_label'] ==  Label]
    unique_audio_ids = filtered_df['audio_id'].unique()
        
    train_audio_ids, val_audio_ids = train_test_split(
    unique_audio_ids, test_size=0.2, random_state=42, shuffle=True)
        
    train_idx = filtered_df[filtered_df['audio_id'].isin(train_audio_ids)].index
    val_idx = filtered_df[filtered_df['audio_id'].isin(val_audio_ids)].index
        
    train_dataset = HDF5Mel_Dataset(HDF5_PATH, train_idx,Label)
    val_dataset = HDF5Mel_Dataset(HDF5_PATH, val_idx,Label)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

    target_df = train_df[train_df['class_label'] == Label]
    num_classes = len(target_df[f"class_name_{name}"].unique())
    model = timm.create_model("efficientnet_b3", in_chans=1, num_classes=num_classes, pretrained=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()   # 單標籤分類用
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0

    #Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)
    
    early_stopper = EarlyStopping(patience=7)
    
    scaler = GradScaler()
    
    for epoch in range(EPOCHS):
        
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            mel = batch['mel'].to(DEVICE)  # (B, 1, 128, T)
            label = batch['class_label'].to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(mel)
                loss = criterion(output, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * mel.size(0)
            _, pred = torch.max(output, 1)
            train_correct += (pred == label).sum().item()
            total += mel.size(0)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/total:.4f} | Train Acc: {train_correct/total:.4f}")
        
        # ----- 驗證 -----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                mel = batch['mel'].to(DEVICE)
                label = batch['class_label'].to(DEVICE)
                output = model(mel)
                loss = criterion(output, label)
                val_loss += loss.item() * mel.size(0)
                _, pred = torch.max(output, 1)
                val_correct += (pred == label).sum().item()
                val_total += mel.size(0)
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")

        # ----- 儲存最佳模型 -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"C:/Users/brad9/Desktop/BirdCLEF/model/best_model_{name}.pth")
            print("Saved new best model!")
            
        scheduler.step(val_acc)
        early_stopper.step(val_acc)
        if early_stopper.early_stop:
            print("Early stopping!")
            break
            
if __name__ == "__main__":
    main()