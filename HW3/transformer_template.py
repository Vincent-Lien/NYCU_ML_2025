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

# ==========================================
# STEP 1: DATASET PREPARATION
# ==========================================
# Students: Unzip and Split the dataset into 70% Train / 30% Test
def prepare_data():
    zip_path = 'dataset.zip'
    extract_path = './dataset'
    target_dir = 'dataset_split'
    
    # TODO: Implement unzipping logic
    
    # TODO: Implement 70/30 split logic into TARGET_DIR
    
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
train_loader = None 
test_loader = None

# ==========================================
# STEP 3: ViT ARCHITECTURE (CORE IMPLEMENTATION)
# ==========================================

# TODO 1: Define PreNorm block
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # TODO: Initialize LayerNorm and store the function (fn)
        pass

    def forward(self, x, **kwargs):
        # TODO: Apply normalization then the function
        pass

# TODO 2: Define FeedForward block
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # TODO: Define two Linear layers with GELU activation and Dropout
        pass

    def forward(self, x):
        pass

# TODO 3: Define Attention block
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        # TODO: Setup qkv projections and softmax attention
        pass

    def forward(self, x):
        # TODO: Implement Scaled Dot-Product Attention using einops rearrange
        pass

# TODO 4: Define Transformer Encoder
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        # TODO: Stack multiple PreNorm + Attention + FeedForward layers
        pass

    def forward(self, x):
        pass

# TODO 5: Define Vision Transformer (ViT)
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, 
                 mlp_dim, pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # TODO: Setup patch embedding, positional embedding, and [CLS] token
        # HINT: Use Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', ...)
        pass

    def forward(self, img):
        # TODO: Apply patch embedding -> add [CLS] -> add Positional Embedding -> Transformer
        pass

# ==========================================
# STEP 4: TRAINING & EVALUATION FUNCTIONS
# ==========================================

# TODO 6: Define training epoch
def train_epoch(model, optimizer, data_loader, loss_history):
    model.train()
    # TODO: Implement standard PyTorch training loop (Zero Grad -> Forward -> Backward -> Step)
    pass

# TODO 7: Define evaluation function
def evaluate(model, data_loader, loss_history):
    model.eval()
    # TODO: Compute test accuracy and average loss
    pass

# ==========================================
# STEP 5: MAIN EXECUTION
# ==========================================
# TODO 8: Initialize model with hyperparameters and set optimizer
model = None
optimizer = None

train_loss_history, test_loss_history = [], []
N_EPOCHS = 1 # Students: Adjust this for hyperparameter tuning!

for epoch in range(1, N_EPOCHS + 1):
    print(f'Epoch: {epoch}')
    train_epoch(model, optimizer, train_loader, train_loss_history)
    evaluate(model, test_loader, test_loss_history)

# TODO 9: Save trained model
# torch.save(model.state_dict(), 'Student_ID_HW3.pth')

# ==========================================
# STEP 6: VISUALIZATION (Requirement c & d)
# ==========================================
# TODO 10: Implement Confusion Matrix and 24-sample Grid plotting
def plot_results():
    # Hint: Use sklearn.metrics.confusion_matrix and seaborn
    pass

# plot_results()