"""
Machine Learning — Homework 4
VAE, GAN, and Denoising Diffusion Probabilistic Model (DDPM)

NOTE: This homework requires PyTorch and torchvision.
      Do NOT use scikit-learn. All model components must be
      implemented using torch.nn — do not call any pretrained
      weights or external model APIs.

Fill in every section marked with:
    # TODO ── <description>
    raise NotImplementedError
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 0. GLOBAL CONFIGURATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 10
lr = 1e-3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train  = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

target_classes  = [9, 0, 1]     # 313581009
mnist_indices   = [i for i, (_, label) in enumerate(mnist_train)  if label in target_classes]
fmnist_indices  = [i for i, (_, label) in enumerate(fmnist_train) if label in target_classes]

mnist_loader  = DataLoader(Subset(mnist_train,  mnist_indices),  batch_size=batch_size, shuffle=True, drop_last=True)
fmnist_loader = DataLoader(Subset(fmnist_train, fmnist_indices), batch_size=batch_size, shuffle=True, drop_last=True)


# ==========================================
# TASK 3: DENOISING DIFFUSION MODEL (DDPM)
# ==========================================

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding — maps scalar timestep t to a continuous vector.

    Architecture: Linear(1, dim) → sin(·)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(1, dim)

    def forward(self, t):
        """
        Args:
            t : (B,) integer timestep tensor
        Returns:
            (B, dim) time embedding
        """
        # TODO ── cast t to float, unsqueeze to (B,1), pass through self.lin, apply sin
        t = t.float().unsqueeze(1)
        return torch.sin(self.lin(t))


class SimpleDiffusionUNet(nn.Module):
    """
    Time-conditioned UNet for DDPM noise prediction.  Eq. 20.2, Algorithm 20.1.

    Architecture:
        inc   : Conv2d(1 → 64,  3×3, padding=1) → ReLU       [28×28]
        down1 : Conv2d(64 → 128, 3×3, stride=2, padding=1) → ReLU  [14×14]
        time_proj : Linear(time_emb_dim, 128)   ← injected into down1 features
        up1   : ConvTranspose2d(128 → 64, 4×4, stride=2, padding=1) → ReLU [28×28]
        outc  : Conv2d(64 → 1, 3×3, padding=1)
    """
    def __init__(self, time_emb_dim=32):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_emb_dim)

        # TODO ── define self.inc, self.down1, self.time_proj, self.up1, self.outc
        self.inc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.time_proj = nn.Linear(time_emb_dim, 128)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.outc = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x : (B, 1, 28, 28) noisy image x_t
            t : (B,) integer timestep tensor
        Returns:
            (B, 1, 28, 28) predicted noise ε̂
        """
        # TODO ── 1. compute t_emb via self.time_mlp(t)
        t_emb = self.time_mlp(t)
        # TODO ── 2. project t_emb to (B, 128, 1, 1) via self.time_proj
        t_proj = self.time_proj(t_emb).view(x.size(0), 128, 1, 1)
        # TODO ── 3. pass x through inc → down1, add time projection
        x1 = self.inc(x)
        x2 = self.down1(x1) + t_proj
        # TODO ── 4. pass through up1 → outc and return
        x3 = self.up1(x2)
        return self.outc(x3)


# --- Fixed diffusion schedule ---
T         = 300
beta      = torch.linspace(1e-4, 0.02, T).to(device)
alpha     = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


def forward_diffusion(x_0, t):
    """
    Closed-form forward diffusion — Eq. 20.2:
        q(z_t | x_0) = N(z_t | √ᾱ_t · x_0, (1 − ᾱ_t) · I)
        x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)

    Args:
        x_0 : (B, 1, 28, 28) clean images
        t   : (B,) integer timestep indices in [0, T)
    Returns:
        x_t        : (B, 1, 28, 28) noisy images
        noise (ε)  : (B, 1, 28, 28) the noise added
    """
    # TODO ── index alpha_bar at t, reshape to (B,1,1,1)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    # TODO ── sample ε ~ N(0,I) with torch.randn_like(x_0)
    noise = torch.randn_like(x_0)
    # TODO ── compute x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
    x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1 - ab) * noise
    # TODO ── return (x_t, noise)
    return x_t, noise


# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
if __name__ == "__main__":
    os.makedirs('./output_plots', exist_ok=True)
    print(f"Running on: {device}\n" + "="*50)

    # ------------------------------------------
    # Task 3: Train DDPM
    # ------------------------------------------
    print("\n" + "="*50 + "\nTask 3: Training DDPM...")
    diffusion_model  = SimpleDiffusionUNet().to(device)
    optimizer_diff   = optim.Adam(diffusion_model.parameters(), lr=lr)
    criterion_diff   = nn.MSELoss()

    diffusion_model.train()
    for epoch in range(1, epochs + 1):
        total_diff_loss = 0
        for images, _ in mnist_loader:
            images = images.to(device)
            b_size = images.size(0)

            t             = torch.randint(0, T, (b_size,), device=device)
            x_t, true_noise = forward_diffusion(images, t)
            predicted_noise  = diffusion_model(x_t, t)
            loss = criterion_diff(predicted_noise, true_noise)

            optimizer_diff.zero_grad()
            loss.backward()
            optimizer_diff.step()
            total_diff_loss += loss.item()
        print(f"DDPM | Epoch [{epoch}/{epochs}] | MSE Loss: {total_diff_loss / len(mnist_loader):.4f}")

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    print("\n" + "="*50 + "\nSaving output plots...")

    # DDPM: Denoising Timeline (6 frames)
    diffusion_model.eval()
    with torch.no_grad():
        num_frames     = 6
        save_intervals = np.linspace(T - 1, 0, num_frames, dtype=int)
        snapshots      = []
        x_seq          = torch.randn(1, 1, 28, 28, device=device)

        for t_idx in reversed(range(0, T)):
            t_tensor  = torch.full((1,), t_idx, device=device, dtype=torch.long)
            pred_eps  = diffusion_model(x_seq, t_tensor)
            beta_t    = beta[t_idx]; alpha_t = alpha[t_idx]; abar_t = alpha_bar[t_idx]
            mean      = (1 / torch.sqrt(alpha_t)) * (x_seq - (beta_t / torch.sqrt(1 - abar_t)) * pred_eps)
            x_seq     = mean + torch.sqrt(beta_t) * torch.randn_like(x_seq) if t_idx > 0 else mean
            if t_idx in save_intervals:
                snapshots.append(((x_seq.clone().squeeze() + 1) / 2).clamp(0, 1).cpu().numpy())

        fig, axes = plt.subplots(1, num_frames, figsize=(15, 3))
        for idx, snap in enumerate(snapshots):
            axes[idx].imshow(snap, cmap='gray')
            axes[idx].set_title(f"Timestep t = {save_intervals[idx]}")
            axes[idx].axis('off')
        plt.suptitle("DDPM Reverse Denoising Timeline", y=1.05); plt.tight_layout()
        # plt.savefig('./output_plots/ddpm_denoising_timeline.png'); plt.close()

    print("\nAll plots saved to './output_plots/'")
