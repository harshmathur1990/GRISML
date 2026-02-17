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

        # --------------------------------------------------
        # optional dynamic-range scaling for other params
        # --------------------------------------------------
        if APPLY_OUTPUT_RESCALE:

            # blocks
            vlos_slice  = slice(ltau, 2*ltau)
            vturb_slice = slice(2*ltau, 3*ltau)
            blong_slice = slice(3*ltau, 4*ltau)

            y[vlos_slice]  *= OUTPUT_MULTIPLIERS["vlos"]
            y[vturb_slice] *= OUTPUT_MULTIPLIERS["vturb"]
            y[blong_slice] *= OUTPUT_MULTIPLIERS["blong"]

        return (
            ca,
            si,
            torch.from_numpy(y)
        )
