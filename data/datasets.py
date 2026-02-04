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
        self.ca_hdul = fits.open(ca_fits, memmap=True)
        self.si_hdul = fits.open(si_fits, memmap=True)

        self.ca = self.ca_hdul[0].data
        self.si = self.si_hdul[0].data

        # ============================
        # NEW: handle missing time dim
        # ============================
        if self.ca.ndim == 4:
            # (s, y, x, λ) → pretend t=0
            self.ca_has_time = False
        elif self.ca.ndim == 5:
            self.ca_has_time = True
        else:
            raise ValueError(f"Unexpected Ca shape: {self.ca.shape}")

        if self.si.ndim == 4:
            self.si_has_time = False
        elif self.si.ndim == 5:
            self.si_has_time = True
        else:
            raise ValueError(f"Unexpected Si shape: {self.si.shape}")

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
        ti_ca = t if self.ca_has_time else 0
        ti_si = t if self.si_has_time else 0

        # ---- input ----
        ca = self.ca[ti_ca, STOKES_IDX, y, x, :] if self.ca_has_time \
             else self.ca[STOKES_IDX, y, x, :]

        si = self.si[ti_si, STOKES_IDX, y, x, :] if self.si_has_time \
             else self.si[STOKES_IDX, y, x, :]

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
