import torch
import torch.nn as nn


# ============================================================
# Basic 1D convolution block
# ============================================================
class ResidualConvBlock1D(nn.Module):
    def __init__(self, cin, cout, k=5, p=2, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv1d(cin, cout, kernel_size=k, padding=p*dilation, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(cout)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(cout, cout, kernel_size=k, padding=p*dilation, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(cout)

        if cin != cout:
            self.skip = nn.Conv1d(cin, cout, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        return self.relu(x)


# ============================================================
# Spectral encoder with wavelength weighting
# ============================================================
class SpectralEncoder(nn.Module):
    def __init__(
        self,
        n_stokes,
        latent_dim,
        wavelength_weight: torch.Tensor,
        stokes_scale: torch.Tensor
    ):
        super().__init__()

        if wavelength_weight.ndim != 1:
            raise ValueError("wavelength_weight must be 1D")

        self.register_buffer("w_lambda", wavelength_weight[None, None, :])
        self.register_buffer("w_stokes", stokes_scale[None, :, :])

        self.conv = nn.Sequential(

            # ---- stage 1 ----
            ResidualConvBlock1D(n_stokes, 64),
            ResidualConvBlock1D(64, 64),
            nn.MaxPool1d(2),

            # ---- stage 2 ----
            ResidualConvBlock1D(64, 128),
            ResidualConvBlock1D(128, 128),
            nn.MaxPool1d(2),

            # ---- stage 3 ----
            ResidualConvBlock1D(128, 256),
            ResidualConvBlock1D(256, 256, dilation=2),
            nn.MaxPool1d(2),

            # ---- stage 4 ----
            ResidualConvBlock1D(256, 512),
            ResidualConvBlock1D(512, 512, dilation=4),

            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        if x.shape[-1] != self.w_lambda.shape[-1]:
            raise ValueError("Lambda mismatch")

        x = x * self.w_stokes
        x = x * self.w_lambda

        x = self.conv(x)
        x = x.squeeze(-1)

        return self.fc(x)


# ============================================================
# Full Ca + Si inversion model
# ============================================================
class CaSiInversionCNN(nn.Module):
    def __init__(
        self,
        n_stokes,
        ltau,
        ca_weight,
        si_weight,
        stokes_scale,
        latent_dim=256
    ):
        super().__init__()

        self.ca_encoder = SpectralEncoder(
            n_stokes=n_stokes,
            latent_dim=latent_dim,
            wavelength_weight=ca_weight,
            stokes_scale=stokes_scale
        )

        self.si_encoder = SpectralEncoder(
            n_stokes=n_stokes,
            latent_dim=latent_dim,
            wavelength_weight=si_weight,
            stokes_scale=stokes_scale
        )

        self.trunk = nn.Sequential(
            nn.Linear(2 * latent_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.head = nn.Linear(512, 4 * ltau)

    def forward(self, ca, si):
        z_ca = self.ca_encoder(ca)
        z_si = self.si_encoder(si)

        z = torch.cat([z_ca, z_si], dim=1)
        z = self.trunk(z)

        return self.head(z)

