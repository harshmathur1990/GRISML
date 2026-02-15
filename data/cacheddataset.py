from torch.utils.data import Dataset
import torch

class CachedDataset(Dataset):
    def __init__(self, Ca, Si, Y):
        self.Ca = Ca
        self.Si = Si
        self.Y  = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.Ca[i]),
            torch.from_numpy(self.Si[i]),
            torch.from_numpy(self.Y[i]),
        )
