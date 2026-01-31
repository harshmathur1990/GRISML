import h5py
from astropy.io import fits
import torch
from torch.utils.data import DataLoader

from config import *
from data.dataset import CaSiAtmosDataset
from utils.split import make_splits
from models.model import CaSiInversionCNN
from losses.loss import AtmosLoss
from utils.early_stopping import EarlyStopping
from tqdm import tqdm


# =========================
#   INSPECT DATA SHAPES
# =========================
with fits.open(CA_FITS, memmap=True) as f:
    t, _, y, x, lca = f[0].data.shape

with fits.open(SI_FITS, memmap=True) as f:
    _, _, _, _, lsi = f[0].data.shape

with h5py.File(ATM_H5, "r") as f:
    ltau = f["temp"].shape[-1]

n_stokes = len(STOKES_IDX)


# =========================
#        SPLITS
# =========================
train_idx, val_idx, test_idx = make_splits(
    (t, y, x),
    TRAIN_SPLIT,
    VAL_SPLIT
)

train_ds = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, train_idx)
val_ds   = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, val_idx)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)


# =========================
#        MODEL
# =========================
device = torch.device(DEVICE)

model = CaSiInversionCNN(
    n_stokes=n_stokes,
    ltau=ltau,
).to(device)

criterion = AtmosLoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)

early_stopping = EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=MIN_DELTA,
    path=CHECKPOINT_PATH,
)


# =========================
#     SANITY CHECK
# =========================
ca, si, y = next(iter(train_loader))
print("Ca:", ca.shape)
print("Si:", si.shape)
print("Y :", y.shape)

with torch.no_grad():
    out = model(ca.to(device), si.to(device))
    print("Out:", out.shape)


# =========================
#       TRAINING
# =========================
for ep in range(EPOCHS):

    # =========================
    #        TRAIN
    # =========================
    model.train()
    train_loss = 0.0

    train_pbar = tqdm(
        train_loader,
        desc=f"Epoch {ep:03d} [train]",
        leave=False,
    )

    for ca, si, y in train_pbar:
        ca = ca.to(device, non_blocking=True)
        si = si.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        pred = model(ca, si)
        loss = criterion(pred, y)

        loss.backward()
        optim.step()

        train_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.2e}")

    train_loss /= len(train_loader)

    # =========================
    #      VALIDATION
    # =========================
    model.eval()
    val_loss = 0.0

    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {ep:03d} [val]  ",
        leave=False,
    )

    with torch.no_grad():
        for ca, si, y in val_pbar:
            ca = ca.to(device, non_blocking=True)
            si = si.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            pred = model(ca, si)
            loss = criterion(pred, y)

            val_loss += loss.item()
            val_pbar.set_postfix(loss=f"{loss.item():.2e}")

    val_loss /= len(val_loader)

    # =========================
    #        LOGGING
    # =========================
    print(
        f"Epoch {ep:03d} | "
        f"Train {train_loss:.3e} | "
        f"Val {val_loss:.3e}"
    )

    # =========================
    #   EARLY STOPPING
    # =========================
    if early_stopping.step(val_loss, model):
        print(
            f"Early stopping at epoch {ep} | "
            f"Best val = {early_stopping.best_loss:.3e}"
        )
        break
