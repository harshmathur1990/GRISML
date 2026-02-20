import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from astropy.io import fits
from config import OUTPUT_SCALES, STOKES_IDX


class CaSiAtmosDataset(Dataset):
    """
    Inputs:
      Ca FITS: (t, s, y, x, λ) OR (s, y, x, λ)
      Si FITS: (t, s, y, x, λ) OR (s, y, x, λ)

    Outputs (HDF5):
      temp, vlos, vturb, blong: (t, y, x, ltau)
      (t may be 1)
    """

    def __init__(self, ca_fits, si_fits, atm_h5, indices):
        self.indices = indices

        # ---- FITS inputs ----
        self.Ca_hdul = fits.open(ca_fits, memmap=True)
        self.Si_hdul = fits.open(si_fits, memmap=True)

        self.Ca = self.Ca_hdul[0].data
        self.Si = self.Si_hdul[0].data

        # ============================
        # NEW: handle missing time dim
        # ============================
        if self.Ca.ndim == 4:
            # (s, y, x, λ) → pretend t=0
            self.Ca_has_time = False
        elif self.Ca.ndim == 5:
            self.Ca_has_time = True
        else:
            raise ValueError(f"Unexpected Ca shape: {self.Ca.shape}")

        if self.Si.ndim == 4:
            self.Si_has_time = False
        elif self.Si.ndim == 5:
            self.Si_has_time = True
        else:
            raise ValueError(f"Unexpected Si shape: {self.Si.shape}")

        # ---- HDF5 outputs ----
        self.fatm = h5py.File(atm_h5, "r")
        self.temp  = self.fatm["temp"]
        self.vlos  = self.fatm["vlos"]
        self.vturb = self.fatm["vturb"]
        self.blong = self.fatm["blong"]

        self.sc = OUTPUT_SCALES

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, y, x = self.indices[idx]

        # ============================
        # NEW: time-safe indexing
        # ============================
        ti_ca = t if self.Ca_has_time else 0
        ti_si = t if self.Si_has_time else 0

        # ---- input ----
        ca = self.Ca[ti_ca, STOKES_IDX, y, x, :] if self.Ca_has_time \
             else self.Ca[STOKES_IDX, y, x, :]

        si = self.Si[ti_si, STOKES_IDX, y, x, :] if self.Si_has_time \
             else self.Si[STOKES_IDX, y, x, :]

        # ---- output (normalised) ----
        Y = np.concatenate([
            self.temp[t, y, x]  * self.sc["temp"],
            self.vlos[t, y, x]  * self.sc["vlos"],
            self.vturb[t, y, x] * self.sc["vturb"],
            self.blong[t, y, x] * self.sc["blong"],
        ])

        ca = np.asarray(ca, dtype=np.float32)
        si = np.asarray(si, dtype=np.float32)
        Y  = np.asarray(Y,  dtype=np.float32)

        return (
            torch.from_numpy(ca),
            torch.from_numpy(si),
            torch.from_numpy(Y),
        )
