import os
import time
import shutil
import random
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import numpy as np

# For reproducibility
random.seed(42)

# ==========================================
# STEP 1: DATASET PREPARATION
# ==========================================
# Students: Unzip and Split the dataset into 70% Train / 30% Test
def prepare_data():
    zip_path = 'dataset.zip'
    extract_path = './dataset'
    target_dir = 'dataset_split'
    
    # TODO: Implement unzipping logic
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    # TODO: Implement 70/30 split logic into TARGET_DIR
    train_path = os.path.join(target_dir, 'train')
    test_path = os.path.join(target_dir, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for class_dir in os.listdir(extract_path):
        class_path = os.path.join(extract_path, class_dir)

        # get all image files and shuffle them
        images = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
        random.shuffle(images)

        # split into 70% train and 30% test
        split_idx = int(len(images) * 0.7)
        train_images = images[:split_idx]
        test_images  = images[split_idx:]

        # create class subdirectories in train and test folders
        train_class_path = os.path.join(train_path, class_dir)
        test_class_path  = os.path.join(test_path,  class_dir)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path,  exist_ok=True)

        # copy images to respective directories
        for img_name in train_images:
            shutil.copy(os.path.join(class_path, img_name),
                        os.path.join(train_class_path, img_name))
        for img_name in test_images:
            shutil.copy(os.path.join(class_path, img_name),
                        os.path.join(test_class_path, img_name))
    
    print("Dataset split complete!")

prepare_data()

# ==========================================
# STEP 2: DATA LOADING & TRANSFORMS
# ==========================================
transform_custom = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# TODO: Initialize ImageFolder datasets and DataLoaders
# Hint: Use TRAIN_PATH and TEST_PATH
TRAIN_PATH = os.path.join('dataset_split', 'train')
TEST_PATH  = os.path.join('dataset_split', 'test')

train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=transform_custom)
test_dataset  = torchvision.datasets.ImageFolder(root=TEST_PATH,  transform=transform_custom)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ==========================================
# STEP 3: ViT ARCHITECTURE (CORE IMPLEMENTATION)
# ==========================================

# TODO 1: Define PreNorm block
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # TODO: Initialize LayerNorm and store the function (fn)
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # TODO: Apply normalization then the function
        return self.fn(self.norm(x), **kwargs)

# TODO 2: Define FeedForward block
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # TODO: Define two Linear layers with GELU activation and Dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# TODO 3: Define Attention block
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        # TODO: Setup qkv projections and softmax attention
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # TODO: Implement Scaled Dot-Product Attention using einops rearrange
        b, n, _ = x.shape
        h = self.heads

        # Project to Q, K, V and split
        qkv = self.to_qkv(x)                                                    # (b, n, 3 * inner_dim)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', three=3, h=h)
        q, k, v = qkv[0], qkv[1], qkv[2]                                        # each: (b, h, n, d)

        # Scaled dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale               # (b, h, n, n)
        attn = self.softmax(dots)

        # Weighted sum over values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)                          # (b, h, n, d)
        out = rearrange(out, 'b h n d -> b n (h d)')                            # (b, n, inner_dim)

        return self.to_out(out)

# TODO 4: Define Transformer Encoder
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        # TODO: Stack multiple PreNorm + Attention + FeedForward layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)   # Residual connection: Attention
            x = x + ff(x)     # Residual connection: FeedForward
        return x

# TODO 5: Define Vision Transformer (ViT)
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, 
                 mlp_dim, pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # TODO: Setup patch embedding, positional embedding, and [CLS] token
        # HINT: Use Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', ...)
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        num_patches = (image_size // patch_size) ** 2
        patch_dim   = channels * patch_size * patch_size

        # Patch Embedding
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )

        # 可學習位置編碼：num_patches + 1 (含 CLS token)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.emb_dropout   = nn.Dropout(emb_dropout)

        # Transformer Encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # TODO: Apply patch embedding -> add [CLS] -> add Positional Embedding -> Transformer
        # 1. Patch Embedding
        x = self.patch_embedding(img)                       # (b, num_patches, dim)
        b, n, _ = x.shape

        # 2. 拼接 CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)       # (b, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)               # (b, num_patches+1, dim)

        # 3. 加入位置編碼
        x = x + self.pos_embedding
        x = self.emb_dropout(x)

        # 4. Transformer Encoder
        x = self.transformer(x)                             # (b, num_patches+1, dim)

        # 5. 取 CLS token 輸出做分類
        cls_output = x[:, 0]                                # (b, dim)
        return self.mlp_head(cls_output)                    # (b, num_classes)

# ==========================================
# STEP 4: TRAINING & EVALUATION FUNCTIONS
# ==========================================

# TODO 6: Define training epoch
def train_epoch(model: ViT, optimizer, data_loader: DataLoader, loss_history):
    model.train()
    # TODO: Implement standard PyTorch training loop (Zero Grad -> Forward -> Backward -> Step)
    criterion = nn.CrossEntropyLoss()

    for data, target in data_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()

    loss_history.append(loss.item())
    return loss.item()

# TODO 7: Define evaluation function
def evaluate(model: ViT, data_loader: DataLoader, loss_history):
    model.eval()
    # TODO: Compute test accuracy and average loss
    criterion = nn.CrossEntropyLoss(reduction='sum')

    total_loss    = 0
    correct       = 0
    total_samples = len(data_loader.dataset)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred        = output.argmax(dim=1)
            correct    += pred.eq(target).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    loss_history.append(avg_loss)

    return avg_loss, correct, total_samples, accuracy

# ==========================================
# STEP 5: MAIN EXECUTION
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO 8: Initialize model with hyperparameters and set optimizer
model = ViT(
    image_size   = 28,
    patch_size   = 2,
    num_classes  = len(train_loader.dataset.classes),
    dim          = 128,
    depth        = 8,
    heads        = 8,
    mlp_dim      = 256,
    channels     = 1,
    dim_head     = 64,
    dropout      = 0.1,
    emb_dropout  = 0.1
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

train_loss_history, test_loss_history = [], []
N_EPOCHS = 30 # Students: Adjust this for hyperparameter tuning!

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
samples_per_epoch = len(train_loader.dataset)
total_samples_all = samples_per_epoch * N_EPOCHS

print(f'Epoch: {N_EPOCHS}')
for epoch in range(1, N_EPOCHS + 1):
    loss = train_epoch(model, optimizer, train_loader, train_loss_history)
    processed = epoch * samples_per_epoch
    percentage = 100. * epoch / N_EPOCHS
    print(f'[{processed:>6}/{total_samples_all} ({percentage:>3.0f}%)]  Loss: {loss:.6f}')
    scheduler.step()

avg_loss, correct, total, accuracy = evaluate(model, test_loader, test_loss_history)
print(f'\nAverage test loss: {avg_loss:.4f}  Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

# TODO 9: Save trained model
torch.save(model.state_dict(), '313581009.pth')
print("Model saved as 313581009.pth")

# ==========================================
# STEP 6: VISUALIZATION (Requirement c & d)
# ==========================================
# TODO 10: Implement Confusion Matrix and 24-sample Grid plotting
def plot_results():
    # Hint: Use sklearn.metrics.confusion_matrix and seaborn
    model.eval()
    class_names = test_loader.dataset.classes

    # ── 收集所有預測結果 ──────────────────────────────────────────
    all_preds, all_labels, all_images = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_images.append(images.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_images = torch.cat(all_images)          # (N, 1, 28, 28)

    # ── 1. 混淆矩陣 ───────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")

    # ── 2. 6 列 x 4 欄 樣本預測圖 ────────────────────────────────
    num_classes       = len(class_names)  # 6
    samples_per_class = 4
    selected_images, selected_preds, selected_labels = [], [], []

    for cls_idx in range(num_classes):
        indices = np.where(all_labels == cls_idx)[0]
        chosen  = indices[:samples_per_class]
        for idx in chosen:
            selected_images.append(all_images[idx])
            selected_preds.append(all_preds[idx])
            selected_labels.append(all_labels[idx])

    # 固定 6 rows x 4 cols
    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(samples_per_class * 2.5, num_classes * 3.0))
    fig.suptitle('Sample Predictions (Green=Correct / Red=Wrong)', fontsize=13, y=1.01)

    for i, (img, pred, label) in enumerate(zip(selected_images, selected_preds, selected_labels)):
        row, col = divmod(i, samples_per_class)
        ax = axes[row][col]

        img_show = img.squeeze().numpy() * 0.5 + 0.5
        ax.imshow(img_show, cmap='gray')
        ax.axis('off')

        color = 'green' if pred == label else 'red'
        ax.set_title(f'True {class_names[label]}\nPred {class_names[pred]}',
                     fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: sample_predictions.png")

    # ── 3. 模型摘要 ───────────────────────────────────────────────
    print("\n" + "="*65)
    print("MODEL SUMMARY")
    print("="*65)
    try:
        summary(model, input_size=(1, 28, 28), device=str(DEVICE))
    except ImportError:
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f"\nTotal Parameters    : {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

plot_results()