import h5py
from astropy.io import fits
import torch
from torch.utils.data import DataLoader

from config import *
from data.dataset import CaSiAtmosDataset
from utils.split import make_splits
from models.model import InversionMLP
from losses.loss import AtmosLoss


# ---- inspect input shapes ----
with fits.open(CA_FITS, memmap=True) as f:
    t, s, y, x, lca = f[0].data.shape

with fits.open(SI_FITS, memmap=True) as f:
    _, _, _, _, lsi = f[0].data.shape

with h5py.File(ATM_H5, "r") as f:
    ltau = f["temp"].shape[-1]

n_input = s * (lca + lsi)
n_output = 4 * ltau


# ---- split ----
train_idx, val_idx, test_idx = make_splits(
    (t, y, x),
    TRAIN_SPLIT,
    VAL_SPLIT
)

train_ds = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, train_idx)
val_ds   = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

device = torch.device(DEVICE)
model = InversionMLP(n_input, n_output).to(device)

criterion = AtmosLoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)


# ---- training ----
for ep in range(EPOCHS):
    model.train()
    tl = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optim.step()
        tl += loss.item()

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            vl += criterion(model(x), y).item()

    print(
        f"Epoch {ep:03d} | "
        f"Train {tl/len(train_loader):.3e} | "
        f"Val {vl/len(val_loader):.3e}"
    )
