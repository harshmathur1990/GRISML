import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from astropy.io import fits
from config import OUTPUT_SCALES, STOKES_IDX


class CaSiAtmosDataset(Dataset):

    def __init__(self, stic_h5, atm_h5, indices):

        print("Opening HDF5 files (stream mode)...")

        fs = h5py.File(stic_h5, "r")
        fa = h5py.File(atm_h5, "r")

        wav = fs["wav"][:]
        profiles = fs["profiles"]

        temp  = fa["temp"]
        vlos  = fa["vlos"]
        vturb = fa["vturb"]
        blong = fa["blong"]

        self.indices = indices
        sc = OUTPUT_SCALES

        # ---- wavelength matching ----
        ca_obs = np.arange(1000)*0.0109907 + 8540.67304823
        si_obs = np.arange(872)*0.0144423 + 10818.6544101

        ca_idx = np.abs(wav[:,None] - ca_obs[None,:]).argmin(axis=0)
        si_idx = np.abs(wav[:,None] - si_obs[None,:]).argmin(axis=0)

        ltau = temp.shape[-1]
        n_out = ltau*4
        N = len(indices)

        print("Building RAM dataset from streamed reads...")
        print("Samples:", N)

        self.Ca = np.empty((N, len(ca_idx), len(STOKES_IDX)), dtype=np.float32)
        self.Si = np.empty((N, len(si_idx), len(STOKES_IDX)), dtype=np.float32)
        self.Y  = np.empty((N, n_out), dtype=np.float32)

        # ---- stream pixel-by-pixel ----
        for i,(t,y,x) in enumerate(indices):

            prof = profiles[t,y,x]

            self.Ca[i] = prof[ca_idx][:,STOKES_IDX]
            self.Si[i] = prof[si_idx][:,STOKES_IDX]

            self.Y[i] = np.concatenate([
                temp[t,y,x]*sc["temp"],
                vlos[t,y,x]*sc["vlos"],
                vturb[t,y,x]*sc["vturb"],
                blong[t,y,x]*sc["blong"],
            ])

            if i % 10000 == 0 and i>0:
                print(f"{i}/{N}")

        fs.close()
        fa.close()

        print("Dataset ready in RAM.")

    def __len__(self):
        return len(self.Ca)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.Ca[i]),
            torch.from_numpy(self.Si[i]),
            torch.from_numpy(self.Y[i]),
        )
