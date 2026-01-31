import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, cin, cout, k=5, p=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, padding=p),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpectralEncoder(nn.Module):
    """
    Input: (B, stokes, lambda)
    Output: (B, latent_dim)
    """

    def __init__(self, n_stokes, latent_dim):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock1D(n_stokes, 32),
            ConvBlock1D(32, 64),
            nn.MaxPool1d(2),

            ConvBlock1D(64, 128),
            ConvBlock1D(128, 128),
            nn.MaxPool1d(2),

            ConvBlock1D(128, 256),
            nn.AdaptiveAvgPool1d(1),  # collapse wavelength
        )

        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x: (B, stokes, lambda)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


class CaSiInversionCNN(nn.Module):
    def __init__(self, n_stokes, ltau, latent_dim=128):
        super().__init__()

        self.ca_encoder = SpectralEncoder(n_stokes, latent_dim)
        self.si_encoder = SpectralEncoder(n_stokes, latent_dim)

        self.trunk = nn.Sequential(
            nn.Linear(2 * latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.head = nn.Linear(512, 4 * ltau)

    def forward(self, ca, si):
        """
        ca, si: (B, stokes, lambda)
        """
        z_ca = self.ca_encoder(ca)
        z_si = self.si_encoder(si)

        z = torch.cat([z_ca, z_si], dim=1)
        z = self.trunk(z)

        return self.head(z)
