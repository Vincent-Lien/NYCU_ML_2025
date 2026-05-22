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

target_classes  = [0, 1, 2]
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
        raise NotImplementedError

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
        # TODO ── 2. compute mu and logvar via fc_mu / fc_logvar
        # TODO ── 3. sample z via reparameterize(mu, logvar)
        # TODO ── 4. decode z and reshape to (B, 1, 28, 28)
        # TODO ── return (recon, mu, logvar)
        raise NotImplementedError


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
    # TODO ── compute kl_loss using Eq. 19.15
    # TODO ── return (recon_loss + beta_weight * kl_loss) / batch_size
    raise NotImplementedError


# ==========================================
# TASK 2: GENERATIVE ADVERSARIAL NETWORK (GAN)
# ==========================================

class Generator(nn.Module):
    """
    GAN Generator — maps z ~ N(0,I) → synthetic image.  Eq. 17.1, 17.8.

    Architecture (Linear MLP with BatchNorm + LeakyReLU, Tanh output):
        z (latent_dim=100)
        → Linear(100, 256) → LeakyReLU(0.2)
        → Linear(256, 512) → BatchNorm1d → LeakyReLU(0.2)
        → Linear(512, 1024) → BatchNorm1d → LeakyReLU(0.2)
        → Linear(1024, 784) → Tanh
        reshape → (B, 1, 28, 28)
    """
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        # TODO ── build self.model as nn.Sequential with the layers above
        raise NotImplementedError

    def forward(self, z):
        """
        Args:
            z : (B, latent_dim) noise vectors
        Returns:
            (B, 1, 28, 28) fake images
        """
        # TODO ── pass z through self.model and reshape output to (B, 1, 28, 28)
        raise NotImplementedError


class Discriminator(nn.Module):
    """
    GAN Discriminator — classifies real (t=1) vs fake (t=0).  Eq. 17.2–17.4.

    Architecture:
        Flatten → Linear(784, 512) → LeakyReLU(0.2)
                → Linear(512, 256) → LeakyReLU(0.2)
                → Linear(256, 1)   → Sigmoid
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO ── build self.model as nn.Sequential with the layers above
        raise NotImplementedError

    def forward(self, img):
        """
        Args:
            img : (B, 1, 28, 28) image tensor
        Returns:
            (B, 1) probability of being real
        """
        # TODO ── pass img through self.model
        raise NotImplementedError


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
        raise NotImplementedError


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
        self.time_mlp  = TimeEmbedding(time_emb_dim)

        # TODO ── define self.inc, self.down1, self.time_proj, self.up1, self.outc
        raise NotImplementedError

    def forward(self, x, t):
        """
        Args:
            x : (B, 1, 28, 28) noisy image x_t
            t : (B,) integer timestep tensor
        Returns:
            (B, 1, 28, 28) predicted noise ε̂
        """
        # TODO ── 1. compute t_emb via self.time_mlp(t)
        # TODO ── 2. project t_emb to (B, 128, 1, 1) via self.time_proj
        # TODO ── 3. pass x through inc → down1, add time projection
        # TODO ── 4. pass through up1 → outc and return
        raise NotImplementedError


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
    # TODO ── sample ε ~ N(0,I) with torch.randn_like(x_0)
    # TODO ── compute x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
    # TODO ── return (x_t, noise)
    raise NotImplementedError


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

    # ------------------------------------------
    # Task 2: Train GAN
    # ------------------------------------------
    print("\n" + "="*50 + "\nTask 2: Training GAN...")
    latent_dim_gan = 100
    netG = Generator(latent_dim=latent_dim_gan).to(device)
    netD = Discriminator().to(device)

    optimizer_G   = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D   = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion_gan = nn.BCELoss()

    netG.train(); netD.train()
    for epoch in range(1, epochs + 1):
        total_loss_D, total_loss_G = 0, 0
        for images, _ in fmnist_loader:
            b_size = images.size(0)
            images = images.to(device)

            label_real = torch.ones(b_size,  1, device=device)
            label_fake = torch.zeros(b_size, 1, device=device)

            # --- Update Discriminator (Eq. 17.7) ---
            output_real  = netD(images)
            loss_D_real  = criterion_gan(output_real, label_real)

            noise        = torch.randn(b_size, latent_dim_gan, device=device)
            fake_images  = netG(noise)
            output_fake  = netD(fake_images.detach())
            loss_D_fake  = criterion_gan(output_fake, label_fake)

            loss_D = loss_D_real + loss_D_fake
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            total_loss_D += loss_D.item()

            # --- Update Generator (Eq. 17.8, non-saturating) ---
            output_g_fake = netD(fake_images)
            loss_G = criterion_gan(output_g_fake, label_real)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            total_loss_G += loss_G.item()

        print(f"GAN | Epoch [{epoch}/{epochs}] | Loss_D: {total_loss_D/len(fmnist_loader):.4f} | Loss_G: {total_loss_G/len(fmnist_loader):.4f}")

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
        axes[0, 0].set_title("Original", loc='left')
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

    # GAN: 4×4 Sample Grid
    netG.eval()
    with torch.no_grad():
        fake_noise       = torch.randn(16, latent_dim_gan, device=device)
        generated_fmnist = netG(fake_noise).cpu()
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(((generated_fmnist[i].squeeze() + 1) / 2).clamp(0, 1), cmap='gray')
            ax.axis('off')
        plt.suptitle("GAN Generated FashionMNIST Samples"); plt.tight_layout()
        plt.savefig('./output_plots/gan_generated_samples.png'); plt.close()

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
            axes[idx].set_title(f"t = {save_intervals[idx]}")
            axes[idx].axis('off')
        plt.suptitle("DDPM Reverse Denoising Timeline", y=1.05); plt.tight_layout()
        plt.savefig('./output_plots/ddpm_denoising_timeline.png'); plt.close()

    print("\nAll plots saved to './output_plots/'")
