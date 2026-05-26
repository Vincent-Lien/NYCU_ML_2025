import torchvision.models as models
import csv
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from torch.utils.data import random_split

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# 1. 定義圖片轉換流程 (例如縮放、轉成 Tensor、標準化)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    # 修改：直接接收切分好的 dataframe，而不是在內部讀取 csv
    def __init__(self, root_dir, labels_csv, gt_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 讀取類別標籤對應表
        self.labels_df = pd.read_csv(labels_csv)
        self.label_to_name = dict(zip(self.labels_df['label_id'], self.labels_df['class_name']))
        
        # 直接使用外部傳入的 dataframe
        self.gt_df = gt_df.reset_index(drop=True)

    def __len__(self):
        return len(self.gt_df)

    def __getitem__(self, idx):
        img_relative_path = self.gt_df.iloc[idx, 0]
        label_id = int(self.gt_df.iloc[idx, 1])
        
        img_full_path = os.path.join(self.root_dir, img_relative_path)
        image = Image.open(img_full_path).convert('RGBA').convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_id


# ==================== 資料分層切分與實例化 ====================

# 1. 在外面先讀取完整的 ground_truth CSV
full_gt_df = pd.read_csv('dataset/ground_truth.csv', header=None, names=['img_path', 'label_id'])

# 2. 使用 stratify 參數進行 8:2 分層切分（確保類別比例一致）
# random_state 可以固定隨機結果，方便日後實驗重現
train_df, val_df = train_test_split(
    full_gt_df, 
    test_size=0.1, 
    stratify=full_gt_df['label_id'], 
    random_state=42
)

# 3. 分別建立訓練與驗證的 Dataset
train_dataset = CustomImageDataset(
    root_dir='dataset',
    labels_csv='dataset/class_labels.csv',
    gt_df=train_df,                  # 傳入 80% 的資料
    transform=data_transforms        # 訓練集通常建議加入 Data Augmentation
)

val_dataset = CustomImageDataset(
    root_dir='dataset',
    labels_csv='dataset/class_labels.csv',
    gt_df=val_df,                    # 傳入 20% 的資料
    transform=data_transforms        # 驗證集通常只做 Resize 和 Normalize
)

# 4. 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# For EfficientNet V2 (Small)
model = models.efficientnet_v2_s(weights='DEFAULT')

# ==========================================
# 4. 配置模型與超參數
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
num_classes = len(train_dataset.labels_df)

num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 10
best_val_acc = 0.0  # 改用「驗證集最高準確率」來篩選最佳模型

# ==========================================
# 5. 開始訓練與驗證迴圈
# ==========================================
for epoch in range(epochs):
    # ------------------
    # 訓練階段 (Train)
    # ------------------
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", unit="batch")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        train_bar.set_postfix(loss=loss.item(), acc=100.0 * train_correct / train_total)
        
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = 100.0 * train_correct / train_total

    # ------------------
    # 驗證階段 (Validation)
    # ------------------
    model.eval()  # 切換為評估模式 (關閉 Dropout 與 BatchNorm 固定)
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    # 驗證時關閉梯度計算，節省記憶體與加速
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val  ", unit="batch")
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            val_bar.set_postfix(loss=loss.item(), acc=100.0 * val_correct / val_total)
            
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = 100.0 * val_correct / val_total
    
    # 印出當前 Epoch 總結
    print(f"\nSummary - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
    print(f"Summary - Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

    # ------------------
    # 6. 儲存模型權重 (.pth)
    # ------------------    
    # 如果驗證集準確率創歷史新高，儲存最佳權重
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"--> 🎉 偵測到更好的驗證準確率！最佳模型已儲存至 'best_model.pth' (Acc: {best_val_acc:.2f}%)")
    print("-" * 50)

print("全部訓練與驗證流程結束！")
