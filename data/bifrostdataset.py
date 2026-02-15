import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from config import OUTPUT_SCALES, STOKES_IDX


class CaSiAtmosDataset(Dataset):
    """
    RAM-cached dataset.

    Loads entire HDF5 cubes into memory once,
    extracts only requested pixels,
    then serves pure NumPy slices (very fast).
    """

    def __init__(self, stic_h5, atm_h5, indices):

        print("Loading HDF5 cubes into RAM...")

        # ---- load STiC cube fully ----
        with h5py.File(stic_h5, "r") as f:
            self.wav = f["wav"][:]
            self.profiles = f["profiles"][:]   # FULL LOAD

        # ---- load atmosphere cube fully ----
        with h5py.File(atm_h5, "r") as f:
            self.temp  = f["temp"][:]
            self.vlos  = f["vlos"][:]
            self.vturb = f["vturb"][:]
            self.blong = f["blong"][:]

        print("Cubes loaded into RAM")

        self.indices = indices
        self.sc = OUTPUT_SCALES

        # ---- observed wavelength grids ----
        ca_obs = np.arange(1000, dtype=float) * 0.0109907 + 8540.67304823
        si_obs = np.arange(872,  dtype=float) * 0.0144423 + 10818.6544101

        # ---- wavelength matching ----
        print("Matching wavelengths...")
        self.ca_idx = np.abs(self.wav[:, None] - ca_obs[None, :]).argmin(axis=0)
        self.si_idx = np.abs(self.wav[:, None] - si_obs[None, :]).argmin(axis=0)

        # ---- determine output size ----
        ltau = self.temp.shape[-1]
        n_out = ltau * 4
        N = len(indices)

        print("Pre-extracting tensors into RAM...")
        print("Samples:", N)

        # ---- allocate arrays ----
        self.Ca = np.empty((N, len(self.ca_idx), len(STOKES_IDX)), dtype=np.float32)
        self.Si = np.empty((N, len(self.si_idx), len(STOKES_IDX)), dtype=np.float32)
        self.Y  = np.empty((N, n_out), dtype=np.float32)

        # ---- extract once ----
        for i, (t, y, x) in enumerate(indices):

            prof = self.profiles[t, y, x]

            self.Ca[i] = prof[self.ca_idx][:, STOKES_IDX]
            self.Si[i] = prof[self.si_idx][:, STOKES_IDX]

            self.Y[i] = np.concatenate([
                self.temp[t, y, x]  * self.sc["temp"],
                self.vlos[t, y, x]  * self.sc["vlos"],
                self.vturb[t, y, x] * self.sc["vturb"],
                self.blong[t, y, x] * self.sc["blong"],
            ])

            if i % 20000 == 0 and i > 0:
                print(f"{i}/{N}")

        print("Dataset ready in RAM.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.Ca[idx]),
            torch.from_numpy(self.Si[idx]),
            torch.from_numpy(self.Y[idx]),
        )
