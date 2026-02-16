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
        ca = torch.from_numpy(self.Ca[i]).permute(1,0)
        si = torch.from_numpy(self.Si[i]).permute(1,0)

        return (
            ca,
            si,
            torch.from_numpy(self.Y[i])
        )
