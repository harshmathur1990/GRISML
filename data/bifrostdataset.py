import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from config import OUTPUT_SCALES, STOKES_IDX


class CaSiAtmosDataset(Dataset):
    """
    Inputs:
      STiC HDF5:
        wav      (nwav_highres,)
        profiles (t, ny, nx, nwav_highres, stokes)

    Outputs (HDF5 atmosphere):
      temp, vlos, vturb, blong: (t, y, x, ltau)
    """

    def __init__(self, stic_h5, atm_h5, indices):
        self.indices = indices

        # ---- STiC synthetic input ----
        self.fsyn = h5py.File(stic_h5, "r")
        self.wav = self.fsyn["wav"][:]                      # (nwav,)
        self.profiles = self.fsyn["profiles"]               # lazy

        # ---- Observed wavelength grids ----
        self.ca_obs_wav = np.arange(1000, dtype=float) * 0.0109907 + 8540.67304823
        self.si_obs_wav = np.arange(872,  dtype=float) * 0.0144423 + 10818.6544101

        # ---- Precompute nearest wavelength indices ----
        self.ca_idx = np.abs(self.wav[:, None] - self.ca_obs_wav[None, :]).argmin(axis=0)
        self.si_idx = np.abs(self.wav[:, None] - self.si_obs_wav[None, :]).argmin(axis=0)

        # ---- HDF5 atmosphere outputs ----
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

        # ---- input from STiC ----
        # profiles shape: (t, ny, nx, wav, stokes)
        prof = self.profiles[t, y, x]   # (wav, stokes)

        ca = prof[self.ca_idx, STOKES_IDX]
        si = prof[self.si_idx, STOKES_IDX]

        # ---- output (normalised atmosphere) ----
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
