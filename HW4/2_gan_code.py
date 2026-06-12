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
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Args:
            z : (B, latent_dim) noise vectors
        Returns:
            (B, 1, 28, 28) fake images
        """
        # TODO ── pass z through self.model and reshape output to (B, 1, 28, 28)
        img = self.model(z)
        return img.view(z.size(0), 1, 28, 28)


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
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        """
        Args:
            img : (B, 1, 28, 28) image tensor
        Returns:
            (B, 1) probability of being real
        """
        # TODO ── pass img through self.model
        return self.model(img)


# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
if __name__ == "__main__":
    os.makedirs('./output_plots', exist_ok=True)
    print(f"Running on: {device}\n" + "="*50)

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
    loss_history_D = []
    loss_history_G = []

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

        print(f"GAN | Epoch [{epoch}/ {epochs}] | Loss_D: {total_loss_D/len(fmnist_loader):.4f} | Loss_G: {total_loss_G/len(fmnist_loader):.4f}")
        loss_history_D.append(total_loss_D / len(fmnist_loader))
        loss_history_G.append(total_loss_G / len(fmnist_loader))


    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    print("\n" + "="*50 + "\nSaving output plots...")

    # GAN: Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history_D, label='Loss_D', linewidth=2)
    plt.plot(range(1, epochs + 1), loss_history_G, label='Loss_G', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./output_plots/gan_loss_curves.png', dpi=300)
    plt.close()

    # GAN: 4×4 Sample Grid
    netG.eval()
    with torch.no_grad():
        fake_noise       = torch.randn(16, latent_dim_gan, device=device)
        generated_fmnist = netG(fake_noise).cpu()
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(((generated_fmnist[i].squeeze() + 1) / 2).clamp(0, 1), cmap='gray')
            ax.axis('off')
        plt.suptitle("GAN Generated FashionMNIST Sample Grid"); plt.tight_layout()
        plt.savefig('./output_plots/gan_generated_samples.png'); plt.close()

    print("\nAll plots saved to './output_plots/'")
