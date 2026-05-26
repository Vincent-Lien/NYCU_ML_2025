import os
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---- 1. 設定信心度門檻（可自由調整） ----
CONFIDENCE_THRESHOLD = 0.70  # 信心度低於 70% 就抓出來人工檢查


# 2. 自訂只讀取資料夾內所有圖片的 Dataset
class UnlabeledDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        self.img_paths = []
        for ext in extensions:
            self.img_paths.extend(list(self.folder_path.glob(ext)))
            self.img_paths.extend(list(self.folder_path.glob(ext.upper())))
        
        self.img_paths = sorted(list(set(self.img_paths)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGBA').convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path.name


# 3. 設定環境與影像轉換
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
num_classes = 25  # ⚠️ 請改成你訓練時的實際類別總數

# 讀取類別名稱字典（這樣 CSV 會直接顯示文字，方便你人工肉眼看）
labels_df = pd.read_csv('dataset/class_labels.csv')
label_to_name = dict(zip(labels_df['label_id'], labels_df['class_name']))

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_folder = '../test'  # ⚠️ 你的測試圖片資料夾
unlabeled_dataset = UnlabeledDataset(folder_path=test_folder, transform=inference_transforms)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)


# 4. 載入模型與權重
model = models.efficientnet_v2_s(weights=None)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_features, num_classes)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()


# 5. 開始批次推論（加入 Softmax 計算信心度）
all_results = []

with torch.no_grad():
    for images, filenames in tqdm(unlabeled_loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        
        # ⚠️ 將 logits 轉成 0~1 的機率分佈
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 取得最高機率（信心度）與對應的類別 ID
        confidences, predicted_labels = probabilities.max(1)
        
        # 轉回 CPU 處理
        confidences = confidences.cpu().numpy()
        predicted_labels = predicted_labels.cpu().numpy()
        
        for fname, conf, pred in zip(filenames, confidences, predicted_labels):
            class_name = label_to_name.get(pred, "Unknown")
            
            all_results.append({
                'img_path': fname,
                'predicted_label_id': int(pred),
                'predicted_class_name': class_name,
                'confidence': float(conf)  # 數值介於 0 ~ 1
            })

# 6. 轉成 DataFrame 並進行篩選
df = pd.DataFrame(all_results)

# 儲存所有圖片的完整預測結果
df.to_csv('all_inference_results.csv', index=False)

# ⚠️ 關鍵：篩選出低於門檻的資料，並依照信心度從最低到最高排序
df_review = df[df['confidence'] < CONFIDENCE_THRESHOLD].sort_values(by='confidence', ascending=True)
df_review.to_csv('needs_review.csv', index=False)

print(f"\n🎉 盲測推論完成！")
print(f"👉 完整結果已儲存至 'all_inference_results.csv'")
print(f"⚠️ 信心度低於 {CONFIDENCE_THRESHOLD*100}% 的圖片有 {len(df_review)} 張，已獨立存入 'needs_review.csv' 供你檢查！")
