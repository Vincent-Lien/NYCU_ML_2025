import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# 1. 基礎設定
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
num_classes = 25  # ⚠️ 請改成你訓練時的實際類別總數

# 2. 影像轉換 (推論不需要 Data Augmentation)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 使用 ImageFolder 自動讀取資料夾結構
# ⚠️ 請將 'dataset/test_classified' 改成你存放類別資料夾的上一層目錄
test_dataset = ImageFolder(root='../train', transform=inference_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. 載入模型架構與權重
model = models.efficientnet_v2_s(weights=None)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_features, num_classes)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# 5. 開始批次推論與計算準確率
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"\n🎉 測試完成！總張數: {total} | 準確率 (Accuracy): {100.0 * correct / total:.2f}%")
