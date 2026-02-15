import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from config import OUTPUT_SCALES, STOKES_IDX, DATA_LOADING_MODE
from tqdm import tqdm


class CaSiAtmosDataset(Dataset):

    def __init__(self, stic_h5, atm_h5, indices):

        self.indices = indices
        self.sc = OUTPUT_SCALES

        print(f"Dataset mode: {DATA_LOADING_MODE}")

        # ----------------------------------------------------
        # OPEN FILES
        # ----------------------------------------------------
        fs = h5py.File(stic_h5, "r")
        fa = h5py.File(atm_h5, "r")

        wav = fs["wav"][:]

        # ----------------------------------------------------
        # wavelength matching
        # ----------------------------------------------------
        ca_obs = np.arange(1000)*0.0109907 + 8540.67304823
        si_obs = np.arange(872)*0.0144423 + 10818.6544101

        ca_idx = np.abs(wav[:,None] - ca_obs[None,:]).argmin(axis=0)
        si_idx = np.abs(wav[:,None] - si_obs[None,:]).argmin(axis=0)

        ltau = fa["temp"].shape[-1]
        n_out = ltau * 4
        N = len(indices)

        print("Allocating RAM dataset...")
        self.Ca = np.empty((N, len(ca_idx), len(STOKES_IDX)), dtype=np.float32)
        self.Si = np.empty((N, len(si_idx), len(STOKES_IDX)), dtype=np.float32)
        self.Y  = np.empty((N, n_out), dtype=np.float32)

        # ====================================================
        # MODE 1 — FULL LOAD INTO RAM
        # ====================================================
        if DATA_LOADING_MODE == "full":

            print("Loading full cubes into RAM...")

            profiles = fs["profiles"][:]      # full load
            temp  = fa["temp"][:]
            vlos  = fa["vlos"][:]
            vturb = fa["vturb"][:]
            blong = fa["blong"][:]

            print("Extracting tensors from RAM cubes...")

            pbar = tqdm(total=N, desc="Extracting", unit="pix")

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

                counter += 1

                # update every 20000
                if counter == 20000:
                    pbar.update(counter)
                    counter = 0

            if counter > 0:
                pbar.update(counter)

            pbar.close()

        # ====================================================
        # MODE 2 — STREAM FROM HDF5
        # ====================================================
        else:

            print("Streaming pixels from HDF5...")

            profiles = fs["profiles"]
            temp  = fa["temp"]
            vlos  = fa["vlos"]
            vturb = fa["vturb"]
            blong = fa["blong"]

            pbar = tqdm(total=N, desc="Extracting", unit="pix")

            for i,(t,y,x) in enumerate(indices):

                prof = profiles[t,y,x]

                self.Ca[i] = prof[ca_idx][:,STOKES_IDX]
                self.Si[i] = prof[si_idx][:,STOKES_IDX]

                self.Y[i] = np.concatenate([
                    temp[t,y,x] * self.sc["temp"],
                    vlos[t,y,x] * self.sc["vlos"],
                    vturb[t,y,x] * self.sc["vturb"],
                    blong[t,y,x] * self.sc["blong"],
                ])

                counter += 1

                # update every 20000
                if counter == 20000:
                    pbar.update(counter)
                    counter = 

            # update remaining items
            if counter > 0:
                pbar.update(counter)

            pbar.close()

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
