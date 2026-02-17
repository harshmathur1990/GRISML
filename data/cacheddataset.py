from torch.utils.data import Dataset
import torch
from config import *
import numpy as np


class CachedDataset(Dataset):
    def __init__(self, Ca, Si, Y, logtemp=False):
        self.Ca = Ca
        self.Si = Si
        self.Y  = Y

        self.logtemp = logtemp

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        ca = torch.from_numpy(self.Ca[i]).permute(1,0)
        si = torch.from_numpy(self.Si[i]).permute(1,0)

        y = self.Y[i].copy()

        # --------------------------------------------------
        # optional log10 temperature transform
        # --------------------------------------------------
        if self.logtemp is True:

            ltau = y.shape[0] // 4  # number of depth points per variable

            temp = y[:ltau] / OUTPUT_SCALES["temp"]   # undo scaling
            temp = np.log10(temp)

            y[:ltau] = temp

        return (
            ca,
            si,
            torch.from_numpy(y)
        )
