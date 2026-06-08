#!/usr/bin/env python3
"""
=============================================================
  Multimodal Taiwanese Food Classification
  Student Template — From Scratch (No Pretrained Models)
=============================================================

DATASET STRUCTURE:
  student_package/
  ├── train/
  │   ├── beef_noodles/       (60 images per class)
  │   ├── bubble_tea/
  │   └── ...                 (25 classes total)
  ├── test/
  │   ├── img_00000.jpg       ← no labels, hidden
  │   └── ...
  ├── test_recipes.csv        ← ingredient hint for every test image
  ├── taiwanese_food_descriptions.csv
  ├── class_labels.csv
  └── sample_submission.csv

YOUR GOAL:
  Build a multimodal classifier FROM SCRATCH.
  - Design your own image encoder (CNN)
  - Design your own text encoder (RNN / LSTM / anything you like)
  - Combine both signals to classify 25 Taiwanese food dishes
  - NO pretrained models allowed (no CLIP, no ResNet weights, no BERT, etc.)

HOW TO RUN:
  pip install torch torchvision pillow tqdm numpy
  python student_train_scratch.py

OUTPUT:
  submission.csv  ← upload this to Kaggle
"""

# ==============================================================
# SECTION 1 — IMPORTS
# ==============================================================
import csv
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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="enable_nested_tensor")


# ==============================================================
# SECTION 2 — CONFIGURATION
# ==============================================================

TRAIN_DIR        = Path("train")
TEST_DIR         = Path("test")
LABELS_CSV       = Path("class_labels.csv")
DESCRIPTIONS_CSV = Path("taiwanese_food_descriptions.csv")
TEST_RECIPES_CSV = Path("test_recipes.csv")   # ingredient hint for every test image
SUBMISSION_PATH  = Path("submission.csv")
MODEL_PATH       = Path("my_model_scratch.pth")

IMAGE_SIZE   = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-5

# Output dimension of your image encoder
# TODO: change this if your CNN outputs a different size
IMG_DIM = 256

# Output dimension of your text encoder (each direction for BiLSTM)
# TODO: change this to match your TextEncoder output
TXT_DIM = 128

MAX_LEN = 64        # maximum number of words to read from a text description

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================
# SECTION 3 — DATA TRANSFORMS
# ==============================================================
# TODO: Define train_transform with augmentation.
#       Suggested steps:
#         1. RandomResizedCrop(IMAGE_SIZE)
#         2. RandomHorizontalFlip()
#         3. ColorJitter(brightness, contrast, saturation, hue)
#         4. ToTensor()
#         5. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225],
)

train_transform = transforms.Compose([
    # TODO: add your transforms here
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    NORMALIZE,
])

# TODO: Define val_transform (no random augmentation — just resize, crop, normalize).
#       Suggested steps:
#         1. Resize(IMAGE_SIZE + 32)
#         2. CenterCrop(IMAGE_SIZE)
#         3. ToTensor()
#         4. Normalize (same as above)

val_transform = transforms.Compose([
    # TODO: add your transforms here
    transforms.Resize(IMAGE_SIZE + 32),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    NORMALIZE
])

# TTA: 8 deterministic transforms applied at inference time
TTA_TRANSFORMS = [
    # 1. center crop (same as val)
    transforms.Compose([transforms.Resize(IMAGE_SIZE + 32), transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(), NORMALIZE]),
    # 2. top-left crop
    transforms.Compose([transforms.Resize(IMAGE_SIZE + 32), transforms.FiveCrop(IMAGE_SIZE), transforms.Lambda(lambda crops: crops[0]), transforms.ToTensor(), NORMALIZE]),
    # 3. top-right crop
    transforms.Compose([transforms.Resize(IMAGE_SIZE + 32), transforms.FiveCrop(IMAGE_SIZE), transforms.Lambda(lambda crops: crops[1]), transforms.ToTensor(), NORMALIZE]),
    # 4. bottom-left crop
    transforms.Compose([transforms.Resize(IMAGE_SIZE + 32), transforms.FiveCrop(IMAGE_SIZE), transforms.Lambda(lambda crops: crops[2]), transforms.ToTensor(), NORMALIZE]),
    # 5. bottom-right crop
    transforms.Compose([transforms.Resize(IMAGE_SIZE + 32), transforms.FiveCrop(IMAGE_SIZE), transforms.Lambda(lambda crops: crops[3]), transforms.ToTensor(), NORMALIZE]),
    # 6. horizontal flip + center crop
    transforms.Compose([transforms.Resize(IMAGE_SIZE + 32), transforms.CenterCrop(IMAGE_SIZE), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), NORMALIZE]),
    # 7. slightly larger crop
    transforms.Compose([transforms.Resize(IMAGE_SIZE + 64), transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(), NORMALIZE]),
    # 8. horizontal flip + slightly larger crop
    transforms.Compose([transforms.Resize(IMAGE_SIZE + 64), transforms.CenterCrop(IMAGE_SIZE), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), NORMALIZE]),
]
TTA_N = len(TTA_TRANSFORMS)


# ==============================================================
# SECTION 4 — TEXT UTILITIES  (provided — no changes needed)
# ==============================================================

def tokenize(text: str) -> list:
    """Splits text into lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(descriptions: dict) -> dict:
    """
    Counts all words across all class descriptions and builds a word → index dict.
    Index 0 is reserved for <PAD> (padding / unknown words).
    """
    counter = Counter()
    for text in descriptions.values():
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0}
    for word in counter:
        vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict, max_len: int) -> torch.Tensor:
    """
    Converts a text string into a fixed-length integer tensor.
    Unknown words map to index 0 (<PAD>). Sequences are truncated or zero-padded.
    """
    ids  = [vocab.get(t, 0) for t in tokenize(text)[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def load_descriptions(path: Path) -> dict:
    """Returns {class_name: 'ingredients description'} for all 25 classes."""
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row["class_name"]] = row["ingredients"] #+ " " + row["description"]
    return d


def get_class_tokens(descriptions: dict, id_to_class: dict, vocab: dict) -> torch.Tensor:
    """
    Builds a (num_classes, MAX_LEN) tensor of token IDs.
    Row i = encoded description for class i.
    """
    ordered = [descriptions[id_to_class[i]] for i in range(len(id_to_class))]
    return torch.stack([encode_text(t, vocab, MAX_LEN) for t in ordered])


def load_class_map(labels_csv: Path):
    """Returns (class_to_id dict, id_to_class dict)."""
    class_to_id = {}
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            class_to_id[row["class_name"]] = int(row["label_id"])
    return class_to_id, {v: k for k, v in class_to_id.items()}


# ==============================================================
# SECTION 5 — DATASET CLASSES  (provided — no changes needed)
# ==============================================================

class MultimodalDataset(Dataset):
    """
    Wraps an ImageFolder dataset and attaches the class text token tensor.
    Returns (image_tensor, token_ids, label).
    """
    def __init__(self, image_dataset, class_tokens: torch.Tensor, is_train=True):
        self.ds = image_dataset
        self.ct = class_tokens
        self.is_train = is_train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, lbl = self.ds[idx]
        tokens = self.ct[lbl].clone()

        if self.is_train:
            valid_indices = (tokens != 0).nonzero(as_tuple=True)[0].tolist()
            if len(valid_indices) > 0:
                num_to_keep = random.randint(1, min(3, len(valid_indices)))
                keep_indices = random.sample(valid_indices, k=num_to_keep)
                new_tokens = torch.zeros_like(tokens)
                for i, k_idx in enumerate(keep_indices):
                    new_tokens[i] = tokens[k_idx]
                tokens = new_tokens

        return img, tokens, lbl


class FlatTestDataset(Dataset):
    """
    Loads all .jpg images from the flat test/ folder.
    Also loads the matching recipe from test_recipes.csv.
    Returns (image_tensor, token_ids, image_id).
    """
    def __init__(self, test_dir: Path, recipes_csv: Path,
                 vocab: dict, transform=None):
        self.paths     = sorted(test_dir.glob("*.jpg"))
        self.transform = transform
        # Load filename → recipe mapping
        recipe_map = {}
        with open(recipes_csv) as f:
            for row in csv.DictReader(f):
                recipe_map[row["filename"]] = row["recipe"]
        # Build parallel list of encoded token tensors
        self.tokens = [
            encode_text(recipe_map.get(p.name, ""), vocab, MAX_LEN)
            for p in self.paths
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p   = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.tokens[idx], p.stem


# ==============================================================
# SECTION 6 — MODEL DEFINITION
# ==============================================================

# ── 6A: Image Encoder ──────────────────────────────────────────────────────────
#
# TODO: Build your own CNN image encoder.
#
#   Requirements:
#     • Input  : (B, 3, 224, 224) — a batch of RGB images
#     • Output : (B, IMG_DIM)     — one feature vector per image
#     • Must be trained entirely from scratch (no loading pretrained weights)
#
#   Suggested architecture (ResNet-18 style, but you can be creative):
#     stem  : Conv(3→64, 7×7, stride=2) → BN → ReLU → MaxPool
#     stage1: two residual blocks (64 channels)
#     stage2: two residual blocks (128 channels, stride=2 to halve spatial size)
#     stage3: two residual blocks (256 channels, stride=2)
#     stage4: two residual blocks (512 channels, stride=2)
#     AdaptiveAvgPool2d(1) → Flatten → (B, 512)
#
#   A residual block looks like:
#     x → Conv → BN → ReLU → Conv → BN → (+skip) → ReLU
#     where skip = Identity if channels match, else Conv(1×1) to match channels
#
#   Hint: you are free to use a simpler architecture (plain CNN without residuals)
#   if you prefer, but residual connections generally train faster and more stably.

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.skip  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if (stride != 1 or in_channels != out_channels) else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define your layers here
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(ResidualBlock(32, 32))
        self.stage2 = nn.Sequential(ResidualBlock(32, 64, stride=2))
        self.stage3 = nn.Sequential(ResidualBlock(64, 128, stride=2))
        self.stage4 = nn.Sequential(ResidualBlock(128, 256, stride=2))
        self.pool   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: pass x through your layers and return (B, IMG_DIM)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return x.flatten(1)


# ── 6B: Text Encoder ───────────────────────────────────────────────────────────
#
# TODO: Build your own text encoder.
#
#   Requirements:
#     • Input  : (B, MAX_LEN) — integer token IDs
#     • Output : (B, TXT_DIM * 2) if bidirectional LSTM, or (B, TXT_DIM) otherwise
#     • Must be trained entirely from scratch
#
#   Suggested architecture (2-layer BiLSTM):
#     Embedding(vocab_size, TXT_DIM)
#     BiLSTM(TXT_DIM, TXT_DIM, num_layers=2, bidirectional=True)
#     Mean-pool over non-padding tokens → (B, TXT_DIM * 2)
#
#   Why mean-pool instead of taking the last hidden state?
#     Padding tokens (index 0) are filled zeros and should not contribute.
#     Masked mean pooling ignores them: sum(outputs * mask) / sum(mask).

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model

        # [CLS] token embedding，學一個可訓練的 class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encoding（ingredients 無序，但加了不會壞）
        self.pos_embedding = nn.Embedding(MAX_LEN + 1, d_model)  # +1 for CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,          # (B, L, D) 格式
            norm_first=True,           # Pre-LN，訓練更穩定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        # 輸出投影到 TXT_DIM * 2，維持跟原本介面一致
        self.proj = nn.Linear(d_model, TXT_DIM * 2)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape

        # 1. Embedding
        x = self.embedding(token_ids)                        # (B, L, d_model)

        # 2. Positional encoding（position 0 留給 CLS）
        positions = torch.arange(1, L + 1, device=token_ids.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)  # (B, L, d_model)

        # 3. 在最前面插入 [CLS] token
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                       # (B, L+1, d_model)

        # 4. Padding mask（True = 忽略該位置）
        #    CLS 永遠不 mask，padding token 才 mask
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=token_ids.device),  # CLS
            (token_ids == 0),                                                # padding
        ], dim=1)                                             # (B, L+1)

        # 5. Transformer
        out = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, L+1, d_model)
        out = self.norm(out)

        # 6. 取 [CLS] 位置的輸出作為 sentence embedding
        cls_feat = out[:, 0, :]                               # (B, d_model)

        return self.proj(cls_feat)                            # (B, TXT_DIM*2)


# ── 6C: Multimodal Classifier ──────────────────────────────────────────────────
#
# TODO: Combine ImageEncoder and TextEncoder into one classifier.
#
#   The simplest fusion strategy:
#     image logits = Linear(IMG_DIM  → num_classes)   applied to image features
#     text  logits = Linear(TXT_DIM*2 → num_classes)  applied to text features
#     final logits = α * image_logits + (1-α) * text_logits
#
#   You are free to try other fusion approaches:
#     • Concatenate image + text features → single Linear head
#     • Cross-attention between image and text
#     • Cosine similarity (see CosineHead in the reference solution)
#
#   forward() must return (image_logits, text_logits) so the training loop
#   can compute a weighted loss on both.

class MultimodalClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.txt_encoder = TextEncoder(vocab_size)
        self.img_head    = nn.Linear(IMG_DIM,      num_classes)  # TODO: adjust dims if needed
        self.txt_head    = nn.Linear(TXT_DIM * 2,  num_classes)  # TODO: adjust if not BiLSTM
        self.drop        = nn.Dropout(0.4)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor):
        img_feat  = self.drop(self.img_encoder(images))    # (B, IMG_DIM)
        txt_feat  = self.txt_encoder(token_ids)            # (B, TXT_DIM*2)
        img_logits = self.img_head(img_feat)               # (B, num_classes)
        txt_logits = self.txt_head(txt_feat)               # (B, num_classes)
        return img_logits, txt_logits


# ==============================================================
# SECTION 7 — TRAINING LOOP
# ==============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    """
    TODO: Implement one training epoch.

    For each batch (images, token_ids, labels):
      1. Move all tensors to DEVICE
      2. Zero the gradients
      3. Forward pass: img_logits, txt_logits = model(images, token_ids)
      4. Loss = criterion(img_logits, labels) + TXT_WEIGHT * criterion(txt_logits, labels)
         (TXT_WEIGHT = 0.3 is a good starting point — image is the primary signal)
      5. Backward pass + optimizer step
      6. Track running loss and accuracy (use img_logits for accuracy)

    Return: (avg_loss, accuracy)
    """
    model.train()
    TXT_WEIGHT = 0.3
    total_loss = correct = total = 0

    for images, token_ids, labels in tqdm(loader, desc="  train", leave=False):
        # TODO: implement training step
        images, token_ids, labels = images.to(DEVICE), token_ids.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        img_logits, txt_logits = model(images, token_ids)
        loss = criterion(img_logits, labels) + TXT_WEIGHT * criterion(txt_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (img_logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def val_one_epoch(model, loader, criterion):
    """
    TODO: Implement one validation epoch (no gradient updates).

    For each batch:
      1. Move tensors to DEVICE
      2. Forward pass
      3. Compute loss on img_logits
      4. Track accuracy

    Return: (avg_loss, accuracy)
    """
    model.eval()
    total_loss = correct = total = 0

    for images, token_ids, labels in loader:
        # TODO: implement validation step
        images, token_ids, labels = images.to(DEVICE), token_ids.to(DEVICE), labels.to(DEVICE)

        img_logits, txt_logits = model(images, token_ids)
        loss = criterion(img_logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (img_logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ==============================================================
# SECTION 8 — GENERATE SUBMISSION
# ==============================================================

@torch.no_grad()
def generate_submission(model, vocab: dict, id_to_class: dict):
    ALPHA_IMG = 0.5
    ALPHA_TXT = 0.5

    model.eval()

    # Build one dataset per TTA transform, all sharing the same token tensors
    base_ds = FlatTestDataset(TEST_DIR, TEST_RECIPES_CSV, vocab, transform=None)
    n = len(base_ds)

    # Accumulate softmax scores across all TTA passes: shape (n, num_classes)
    num_classes = len(id_to_class)
    accum_scores = torch.zeros(n, num_classes)  # stays on CPU

    for t_idx, tfm in enumerate(TTA_TRANSFORMS):
        # Temporarily swap transform
        base_ds.transform = tfm
        loader = DataLoader(base_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        offset = 0
        for images, token_ids, _ in tqdm(loader, desc=f"  TTA {t_idx+1}/{TTA_N}", leave=False):
            images, token_ids = images.to(DEVICE), token_ids.to(DEVICE)
            img_logits, txt_logits = model(images, token_ids)
            score = (ALPHA_IMG * F.softmax(img_logits, dim=1)
                   + ALPHA_TXT * F.softmax(txt_logits, dim=1)).cpu()
            bs = score.size(0)
            accum_scores[offset:offset + bs] += score
            offset += bs

    # Final prediction: argmax of averaged scores
    pred_ids = accum_scores.argmax(dim=1).tolist()

    # Collect image ids in order
    image_ids = [p.stem for p in base_ds.paths]
    rows = [[img_id + ".jpg", id_to_class[pred_id]]
            for img_id, pred_id in zip(image_ids, pred_ids)]
    rows.sort(key=lambda r: r[0])

    with open(SUBMISSION_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"\n  Saved {len(rows)} predictions → {SUBMISSION_PATH}")


# ==============================================================
# SECTION 9 — MAIN
# ==============================================================

def main():
    print(f"\n{'='*60}")
    print(f"  From-Scratch Multimodal Taiwanese Food Classifier")
    print(f"  Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    # ── Labels and text ───────────────────────────────────────────────────────
    class_to_id, id_to_class = load_class_map(LABELS_CSV)
    num_classes  = len(class_to_id)
    descriptions = load_descriptions(DESCRIPTIONS_CSV)
    vocab        = build_vocab(descriptions)
    class_tokens = get_class_tokens(descriptions, id_to_class, vocab)

    print(f"  Classes : {num_classes}   Vocab size : {len(vocab)}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds     = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds_src   = ImageFolder(TRAIN_DIR, transform=val_transform)

    # TODO: Remap ImageFolder's auto-assigned label IDs to match class_labels.csv
    #       Hint: build folder_map = {cls: class_to_id[cls] for cls in train_ds.classes}
    #             then update train_ds.targets and train_ds.samples:
    #               train_ds.targets = [folder_map[train_ds.classes[t]] for t in train_ds.targets]
    #               train_ds.samples = [(p, folder_map[train_ds.classes[t]]) for p, t in train_ds.samples]
    #             Do the same for val_ds_src.
    folder_map = {cls: class_to_id[cls] for cls in train_ds.classes}
    for ds in (train_ds, val_ds_src):
        ds.targets = [folder_map[ds.classes[t]] for t in ds.targets]
        ds.samples = [(p, folder_map[ds.classes[t]]) for p, t in ds.samples]

    # TODO: Split into train and validation sets.
    #       Suggestion: hold out the last 10 images per class as validation.
    #         labels  = np.array(train_ds.targets)
    #         tr_idx, vl_idx = [], []
    #         for cls in np.unique(labels):
    #             idx = np.where(labels == cls)[0].tolist()
    #             vl_idx.extend(idx[-10:])
    #             tr_idx.extend(idx[:-10])
    labels = np.array(train_ds.targets)
    tr_idx, vl_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0].tolist()
        vl_idx.extend(idx[-10:])
        tr_idx.extend(idx[:-10])

    # TODO: Wrap with MultimodalDataset to attach text tokens:
    #         train_mm = MultimodalDataset(Subset(train_ds,   tr_idx), class_tokens)
    #         val_mm   = MultimodalDataset(Subset(val_ds_src, vl_idx), class_tokens)
    train_mm = MultimodalDataset(Subset(train_ds,   tr_idx), class_tokens, is_train=True)
    val_mm   = MultimodalDataset(Subset(val_ds_src, vl_idx), class_tokens, is_train=False)

    # TODO: Create DataLoaders:
    #         train_loader = DataLoader(train_mm, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    #         val_loader   = DataLoader(val_mm,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    train_loader = DataLoader(train_mm, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)    # TODO: replace
    val_loader   = DataLoader(val_mm,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)    # TODO: replace

    print(f"  (fill in dataset sizes after completing the TODO above)\n")

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    # TODO: Instantiate MultimodalClassifier and move to DEVICE
    #         model = MultimodalClassifier(len(vocab), num_classes).to(DEVICE)
    model = MultimodalClassifier(len(vocab), num_classes).to(DEVICE)   # TODO: replace

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # TODO: Define optimizer — optimize all model parameters
    #         optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)   # TODO: replace

    # TODO (optional): Define a learning rate scheduler for better convergence
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, train_accs, val_accs = [], [], []
    best_val_acc = 0.0
    print(f"  Training for {NUM_EPOCHS} epochs...")
    print(f"  {'Epoch':<8} {'Loss':<10} {'Train Acc':<13} {'Val Acc'}")
    print(f"  {'-'*45}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        _,          val_acc   = val_one_epoch(model, val_loader, criterion)

        # TODO (optional): scheduler.step()
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " ✓"

        print(f"  {epoch:02d}/{NUM_EPOCHS}   {train_loss:.4f}     {train_acc:.3f}         {val_acc:.3f}{marker}")
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    print(f"\n  Best val accuracy : {best_val_acc:.1%}")
    print(f"  Best model saved  → {MODEL_PATH}")
    
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, label='train')
    ax1.set_title('Loss'); ax1.legend()
    ax2.plot(epochs, train_accs, label='train'); ax2.plot(epochs, val_accs, label='val')
    ax2.set_title('Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\n  Running inference on test set...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generate_submission(model, vocab, id_to_class)

    print(f"\n{'='*60}")
    print(f"  Done! Upload submission.csv to Kaggle.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()