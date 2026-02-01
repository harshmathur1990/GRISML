import h5py
from astropy.io import fits
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from data.datasets import CaSiAtmosDataset
from utils.split import make_splits
from models.model import CaSiInversionCNN
from losses.loss import AtmosLoss
from utils.early_stopping import EarlyStopping
from utils.spectral_weights import build_weight_mask



# ============================================================
# 0. PYTORCH / ROCm INFO
# ============================================================
print("PyTorch version:", torch.__version__)
print("ROCm available :", torch.version.hip is not None)
print("CUDA available :", torch.cuda.is_available())


# ============================================================
# 1. GPU SELECTION + NAME  (POINT 2)
# ============================================================
device = torch.device(DEVICE)

if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")


# ============================================================
# 2. GPU MEMORY HELPER  (POINT 3)
# ============================================================
def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(
            f"[{tag}] GPU memory | "
            f"allocated={alloc:.0f} MB | "
            f"reserved={reserved:.0f} MB | "
            f"total={total:.0f} MB"
        )


# ============================================================
# 3. INSPECT DATA SHAPES
# ============================================================
with fits.open(CA_FITS, memmap=True) as f:
    t, _, y, x, lca = f[0].data.shape

with fits.open(SI_FITS, memmap=True) as f:
    _, _, _, _, lsi = f[0].data.shape

with h5py.File(ATM_H5, "r") as f:
    ltau = f["temp"].shape[-1]

n_stokes = len(STOKES_IDX)


# ============================================================
# 4. SPLITS
# ============================================================
train_idx, val_idx, test_idx = make_splits(
    (t, y, x),
    TRAIN_SPLIT,
    VAL_SPLIT
)

train_ds = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, train_idx)
val_ds   = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, val_idx)
test_ds = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, test_idx)

# ============================================================
# 5. SAFE BATCH-SIZE PROBING (OPTIONAL BUT RECOMMENDED)
# ============================================================
def try_batch_size(batch_size):
    try:
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        ca, si, y = next(iter(loader))
        ca = ca.to(device)
        si = si.to(device)
        y  = y.to(device)

        with torch.no_grad():
            _ = model(ca, si)

        print(f"Batch size {batch_size}: OK")
        print_gpu_memory(f"batch={batch_size}")
        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Batch size {batch_size}: OOM")
            torch.cuda.empty_cache()
            return False
        else:
            raise e


# ============================================================
# 6. DATALOADERS (FINAL)
# ============================================================
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

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,      # IMPORTANT
    pin_memory=True,
)


# ============================================================
# 7. MODEL
# ============================================================
ca_weight = build_weight_mask(
    n_lambda=CA_N_WAVELENGTH,
    core_range=CA_CORE_RANGE,
    core_weight=CORE_WEIGHT,
    wing_weight=WING_WEIGHT,
)

si_weight = build_weight_mask(
    n_lambda=SI_N_WAVELENGTH,
    core_range=SI_CORE_RANGE,
    ignore_range=SI_IGNORE_RANGE,
    core_weight=CORE_WEIGHT,
    wing_weight=WING_WEIGHT,
    ignore_weight=IGNORE_WEIGHT,
)

model = CaSiInversionCNN(
    n_stokes=n_stokes,
    ltau=ltau,
    ca_weight=ca_weight,
    si_weight=si_weight,
).to(device)

criterion = AtmosLoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)

early_stopping = EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=MIN_DELTA,
    path=CHECKPOINT_PATH,
)

print_gpu_memory("after model init")


# ============================================================
# 8. SANITY CHECK + DEVICE CHECK  (POINT 4)
# ============================================================
ca, si, y = next(iter(train_loader))
print("Ca shape:", ca.shape)
print("Si shape:", si.shape)
print("Y  shape:", y.shape)

ca = ca.to(device)
si = si.to(device)
y  = y.to(device)

print("ca device   :", ca.device)
print("model device:", next(model.parameters()).device)

with torch.no_grad():
    out = model(ca, si)
    print("Out shape:", out.shape)

print_gpu_memory("after first forward")


# ============================================================
# 9. TRAINING
# ============================================================
for ep in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------
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

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_loss = 0.0

    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {ep:03d} [val]",
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

    # -------------------------
    # LOGGING
    # -------------------------
    print(
        f"Epoch {ep:03d} | "
        f"Train {train_loss:.3e} | "
        f"Val {val_loss:.3e}"
    )

    print_gpu_memory(f"epoch {ep}")

    # -------------------------
    # EARLY STOPPING
    # -------------------------
    if early_stopping.step(val_loss, model):
        print(
            f"Early stopping at epoch {ep} | "
            f"Best val = {early_stopping.best_loss:.3e}"
        )
        break

print("Loading best model for test evaluation")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

test_loss = 0.0

with torch.no_grad():
    for ca, si, y in tqdm(test_loader, desc="Test", leave=False):
        ca = ca.to(device, non_blocking=True)
        si = si.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        pred = model(ca, si)
        loss = criterion(pred, y)

        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"FINAL TEST LOSS (unbiased): {test_loss:.3e}")
