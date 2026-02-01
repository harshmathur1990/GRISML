import torch
import torch.nn as nn


# ============================================================
# Helper: build wavelength weight mask
# ============================================================
def build_weight_mask(
    n_lambda: int,
    core_range=None,        # tuple (start, end)
    ignore_range=None,      # tuple (start, end)
    core_weight: float = 3.0,
    wing_weight: float = 1.0,
    ignore_weight: float = 0.0,
) -> torch.Tensor:
    """
    Returns a tensor of shape (n_lambda,) containing wavelength weights.
    """
    w = torch.full((n_lambda,), wing_weight, dtype=torch.float32)

    if core_range is not None:
        a, b = core_range
        w[a:b] = core_weight

    if ignore_range is not None:
        a, b = ignore_range
        w[a:b] = ignore_weight

    return w


# ============================================================
# Basic 1D convolution block
# ============================================================
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


# ============================================================
# Spectral encoder with wavelength weighting
# ============================================================
class SpectralEncoder(nn.Module):
    """
    Input : (B, stokes, lambda)
    Output: (B, latent_dim)

    Applies wavelength weighting/masking BEFORE convolution.
    """

    def __init__(self, n_stokes, latent_dim, wavelength_weight: torch.Tensor):
        super().__init__()

        if wavelength_weight.ndim != 1:
            raise ValueError("wavelength_weight must be 1D (n_lambda,)")

        # Register buffer so it:
        # - moves with .to(device)
        # - is saved in state_dict
        # - is not trainable
        self.register_buffer(
            "w_lambda",
            wavelength_weight[None, None, :],   # (1, 1, lambda)
        )

        self.conv = nn.Sequential(
            ConvBlock1D(n_stokes, 32),
            ConvBlock1D(32, 64),
            nn.MaxPool1d(2),

            ConvBlock1D(64, 128),
            ConvBlock1D(128, 128),
            nn.MaxPool1d(2),

            ConvBlock1D(128, 256),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x: (B, stokes, lambda)
        if x.shape[-1] != self.w_lambda.shape[-1]:
            raise ValueError(
                f"Lambda mismatch: input {x.shape[-1]} vs mask {self.w_lambda.shape[-1]}"
            )

        # Apply wavelength weighting
        x = x * self.w_lambda

        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


# ============================================================
# Full Ca + Si inversion model
# ============================================================
class CaSiInversionCNN(nn.Module):
    """
    Pixel-wise inversion CNN with:
      - Ca II encoder
      - Si I encoder
      - Config-driven wavelength weighting
    """

    def __init__(self, n_stokes, ltau, latent_dim=128):
        super().__init__()

        # Import physics config here (NOT hardcoded)
        from config import (
            CA_N_WAVELENGTH, CA_CORE_RANGE,
            SI_N_WAVELENGTH, SI_CORE_RANGE, SI_IGNORE_RANGE,
            CORE_WEIGHT, WING_WEIGHT, IGNORE_WEIGHT,
        )

        # Build wavelength masks
        ca_weight = build_weight_mask(
            n_lambda=CA_N_WAVELENGTH,
            core_range=CA_CORE_RANGE,
            core_weight=CORE_WEIGHT,
            wing_weight=WING_WEIGHT,
        )

        si_weight = build_weight_mask(
            n_lambda=SI_N_WAVELENGTH,
            core_range=SI_CORE_RANGE,
            ignore_range=SI_IGNORE_RANGE,
            core_weight=CORE_WEIGHT,
            wing_weight=WING_WEIGHT,
            ignore_weight=IGNORE_WEIGHT,
        )

        # Encoders
        self.ca_encoder = SpectralEncoder(
            n_stokes=n_stokes,
            latent_dim=latent_dim,
            wavelength_weight=ca_weight,
        )

        self.si_encoder = SpectralEncoder(
            n_stokes=n_stokes,
            latent_dim=latent_dim,
            wavelength_weight=si_weight,
        )

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(2 * latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Atmosphere head
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
