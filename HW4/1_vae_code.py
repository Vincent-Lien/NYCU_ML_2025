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
# TASK 1: VARIATIONAL AUTOENCODER (VAE)
# ==========================================

class VAE(nn.Module):
    """
    Variational Autoencoder — Eq. 19.4 (Bishop 2024).

    Encoder: x → (μ, log σ²)   [Eq. 19.13]
    Decoder: z → x̂
    """
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # --- Encoder (provided) ---
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu     = nn.Linear(256, latent_dim)   # outputs μ
        self.fc_logvar = nn.Linear(256, latent_dim)   # outputs log σ²

        # --- Decoder (provided) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick — Eq. 19.17–19.18:
            z = μ + σ ⊙ ε,   ε ~ N(0, I)
            where σ = exp(0.5 * log σ²)

        Args:
            mu     : (B, latent_dim) mean vector
            logvar : (B, latent_dim) log-variance vector
        Returns:
            z      : (B, latent_dim) sampled latent vector
        """
        # TODO ── compute std from logvar, sample ε, return z
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)
        z = mu + std * epsilon
        return z

    def forward(self, x):
        """
        Full VAE forward pass: encode → reparameterize → decode.

        Args:
            x : (B, 1, 28, 28) input images
        Returns:
            recon  : (B, 1, 28, 28) reconstructed images
            mu     : (B, latent_dim)
            logvar : (B, latent_dim)
        """
        # TODO ── 1. pass x through self.encoder
        hidden = self.encoder(x)
        # TODO ── 2. compute mu and logvar via fc_mu / fc_logvar
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        # TODO ── 3. sample z via reparameterize(mu, logvar)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        # TODO ── 4. decode z and reshape to (B, 1, 28, 28)
        recon = decoded.view(x.size(0), 1, 28, 28)
        # TODO ── return (recon, mu, logvar)
        return recon, mu, logvar


def vae_loss_fn(recon_x, x, mu, logvar, beta_weight=1.0):
    """
    ELBO loss — Eq. 19.19:
        L = (1/N) * (L_recon + β · D_KL)

    Reconstruction term (MSE, reduction='sum'):
        L_recon = ||x - x̂||²

    Analytical KL divergence — Eq. 19.15:
        D_KL = -0.5 * Σ_j (1 + log σ²_j - μ²_j - σ²_j)

    Args:
        recon_x     : (B, 1, 28, 28) reconstructed output
        x           : (B, 1, 28, 28) original input
        mu          : (B, latent_dim)
        logvar      : (B, latent_dim)
        beta_weight : scalar β for KL weighting
    Returns:
        scalar loss (mean over batch)
    """
    # TODO ── compute recon_loss using nn.functional.mse_loss with reduction='sum'
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # TODO ── compute kl_loss using Eq. 19.15
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    # TODO ── return (recon_loss + beta_weight * kl_loss) / batch_size
    return (recon_loss + beta_weight * kl_loss) / x.size(0)


# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
if __name__ == "__main__":
    os.makedirs('./output_plots', exist_ok=True)
    print(f"Running on: {device}\n" + "="*50)

    # ------------------------------------------
    # Task 1: Train VAE
    # ------------------------------------------
    print("\nTask 1: Training VAE...")
    vae_model    = VAE(latent_dim=20).to(device)
    optimizer_vae = optim.Adam(vae_model.parameters(), lr=lr)

    vae_model.train()
    for epoch in range(1, epochs + 1):
        total_vae_loss = 0
        for images, _ in mnist_loader:
            images = images.to(device)
            recon, mu, logvar = vae_model(images)
            loss = vae_loss_fn(recon, images, mu, logvar)
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()
            total_vae_loss += loss.item()
        print(f"VAE | Epoch [{epoch}/{epochs}] | ELBO Loss: {total_vae_loss / len(mnist_loader):.4f}")


    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    print("\n" + "="*50 + "\nSaving output plots...")

    # VAE: Reconstructions
    vae_model.eval()
    with torch.no_grad():
        real_imgs, _ = next(iter(mnist_loader))
        real_imgs    = real_imgs.to(device)
        recon_imgs, _, _ = vae_model(real_imgs)

        fig, axes = plt.subplots(2, 8, figsize=(12, 4))
        for i in range(8):
            axes[0, i].imshow(real_imgs[i].cpu().squeeze(), cmap='gray');  axes[0, i].axis('off')
            axes[1, i].imshow(recon_imgs[i].cpu().squeeze(), cmap='gray'); axes[1, i].axis('off')
        axes[0, 0].set_title("Original Real", loc='left')
        axes[1, 0].set_title("VAE Recon", loc='left')
        plt.tight_layout()
        plt.savefig('./output_plots/vae_reconstructions.png'); plt.close()

        # VAE: 2D Latent Manifold
        grid_size = 10
        grid_x    = np.linspace(-2, 2, grid_size)
        grid_y    = np.linspace(-2, 2, grid_size)
        canvas    = np.empty((28 * grid_size, 28 * grid_size))
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample    = torch.zeros(1, vae_model.latent_dim, device=device)
                z_sample[0, 0] = xi; z_sample[0, 1] = yi
                img = ((vae_model.decoder(z_sample).view(28, 28) + 1) / 2).cpu().numpy()
                canvas[(grid_size - i - 1) * 28:(grid_size - i) * 28, j * 28:(j + 1) * 28] = img
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap='gray'); plt.axis('off')
        plt.title("VAE 2D Latent Space Manifold"); plt.tight_layout()
        plt.savefig('./output_plots/vae_manifold_grid.png'); plt.close()

    print("\nAll plots saved to './output_plots/'")
