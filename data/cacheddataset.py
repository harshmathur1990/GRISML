from torch.utils.data import Dataset
import torch
from config import *
import numpy as np


class CachedDataset(Dataset):
    def __init__(self, Ca, Si, Y, logtemp=False):
        self.Ca = Ca
        self.Si = Si

        # copy targets so cache arrays are not modified on disk
        self.Y = Y.copy()

        self.logtemp = logtemp

        # --------------------------------------------------
        # apply log10 temperature ONCE at init
        # --------------------------------------------------
        if self.logtemp:
            print("Applying log10 transform to temperature targets...")

            ltau = self.Y.shape[1] // 4

            # undo scaling
            temp = self.Y[:, :ltau] / OUTPUT_SCALES["temp"]

            # numerical safety (important!)
            temp = np.clip(temp, 1e-6, None)

            # log10 transform
            temp = np.log10(temp)

            # write back
            self.Y[:, :ltau] = temp

            print(
                "Temp(log10) range:",
                np.nanmin(temp),
                np.nanmax(temp)
            )

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        ca = torch.from_numpy(self.Ca[i]).permute(1,0)
        si = torch.from_numpy(self.Si[i]).permute(1,0)

        return (
            ca,
            si,
            torch.from_numpy(self.Y[i])
        )
