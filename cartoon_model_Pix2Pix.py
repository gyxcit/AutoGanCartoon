import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# -----------------------------
# 1. Définir les blocs U-Net
# -----------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs, padding_mode="reflect") if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if use_act else nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

# -----------------------------
# 2. Construire le générateur (U-Net)
# -----------------------------
class Generator(nn.Module):
    def __init__(self, in_channels, features=64):
        super().__init__()
        self.encoder = nn.ModuleList([
            UNetBlock(in_channels, features, kernel_size=4, stride=2, padding=1, down=True, use_act=False),
            UNetBlock(features, features * 2, kernel_size=4, stride=2, padding=1, down=True),
            UNetBlock(features * 2, features * 4, kernel_size=4, stride=2, padding=1, down=True),
        ])

        self.decoder = nn.ModuleList([
            UNetBlock(features * 4, features * 2, kernel_size=4, stride=2, padding=1, down=False),
            UNetBlock(features * 2, features, kernel_size=4, stride=2, padding=1, down=False),
            nn.ConvTranspose2d(features, in_channels, kernel_size=4, stride=2, padding=1),
        ])

    def forward(self, x):
        enc_outputs = []
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)

        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            if idx < len(enc_outputs):
                x = torch.cat([x, enc_outputs[-(idx + 1)]], dim=1)

        return x

# -----------------------------
# 3. Construire le discriminateur
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels, features=64):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels * 2, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x, y):
        return self.disc(torch.cat([x, y], dim=1))

# -----------------------------
# 4. Entraînement du modèle
# -----------------------------
def train(generator, discriminator, loader, opt_g, opt_d, criterion, device):
    for epoch in range(10):  # Nombre d'époques
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # Entraîner le discriminateur
            fake = generator(x)
            disc_real = discriminator(x, y)
            disc_fake = discriminator(x, fake.detach())
            loss_d = (criterion(disc_real, torch.ones_like(disc_real)) + criterion(disc_fake, torch.zeros_like(disc_fake))) / 2

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Entraîner le générateur
            disc_fake = discriminator(x, fake)
            loss_g = criterion(disc_fake, torch.ones_like(disc_fake)) + criterion(fake, y)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            if idx % 10 == 0:
                print(f"Epoch [{epoch}/{10}] Batch {idx}/{len(loader)} Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

# -----------------------------
# 5. Charger les données et exécuter l'entraînement
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.ImageFolder(root="dataset/cartoon", transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    gen = Generator(in_channels=3).to(device)
    disc = Discriminator(in_channels=3).to(device)
    opt_g = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    train(gen, disc, loader, opt_g, opt_d, criterion, device)
