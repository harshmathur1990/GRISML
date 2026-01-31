import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from astropy.io import fits
from config import OUTPUT_SCALES, STOKES_IDX


class CaSiAtmosDataset(Dataset):
    """
    Inputs:
      Ca FITS: (t, s, y, x, λ_ca)
      Si FITS: (t, s, y, x, λ_si)

    Outputs (HDF5):
      temp, vlos, vturb, blong: (t, y, x, ltau)
    """

    def __init__(self, ca_fits, si_fits, atm_h5, indices):
        self.indices = indices

        # ---- FITS inputs ----
        self.ca_hdul = fits.open(ca_fits, memmap=True)
        self.si_hdul = fits.open(si_fits, memmap=True)

        self.ca = self.ca_hdul[0].data   # ndarray
        self.si = self.si_hdul[0].data

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

        # ---- input ----
        ca = self.ca[t, STOKES_IDX, y, x, :]   # (len(STOKES_IDX), λ_ca)
        si = self.si[t, STOKES_IDX, y, x, :]   # (len(STOKES_IDX), λ_si)

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
