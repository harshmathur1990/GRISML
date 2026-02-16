import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"
import h5py
from astropy.io import fits
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config import *
# from data.datasets import CaSiAtmosDataset
from data.bifrostdataset import CaSiAtmosDataset
from data.cacheddataset import CachedDataset
from utils.split import make_splits
from models.model import CaSiInversionCNN
from losses.loss import AtmosLoss
from utils.early_stopping import EarlyStopping
from utils.spectral_weights import build_weight_mask
from utils.stokes_scaling import build_stokes_scale
from torch.utils.data import Subset
from utils.cache import save_dataset_cache, load_dataset_cache


# ============================================================
# 0. PYTORCH / ROCm INFO
# ============================================================
print("PyTorch version:", torch.__version__)
print("ROCm available :", torch.version.hip is not None)
print("CUDA available :", torch.cuda.is_available())



# ============================================================
# GPU MODE DETECTION (single vs multi)
# ============================================================
def configure_gpu_mode():
    if not torch.cuda.is_available():
        return "cpu", 0

    n_gpu = torch.cuda.device_count()
    print(f"Detected {n_gpu} GPU(s)")

    if n_gpu == 1:
        return "single", 1

    # interactive choice
    ans = input("Use multiple GPUs? [y/N]: ").strip().lower()
    if ans == "y":
        return "multi", n_gpu
    else:
        return "single", 1


gpu_mode, n_gpu = configure_gpu_mode()


# ============================================================
# DEVICE SELECTION
# ============================================================
if gpu_mode == "cpu":
    device = torch.device("cpu")
    print("Using CPU")

elif gpu_mode == "single":
    device = torch.device("cuda:0")
    print("Using GPU:", torch.cuda.get_device_name(0))

else:
    device = torch.device("cuda:0")
    print(f"Using {n_gpu} GPUs via DataParallel")


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
# 3. INSPECT DATA SHAPES (TIME-SAFE)
# ============================================================

# ---- atmosphere defines (t, y, x) ----
with h5py.File(ATM_H5, "r") as f:
    _, _, _, ltau = f["temp"].shape

# # ---- Ca FITS ----
# with fits.open(CA_FITS, memmap=True) as f:
#     ca_shape = f[0].data.shape

# if len(ca_shape) == 5:
#     _, _, _, _, lca = ca_shape
# elif len(ca_shape) == 4:
#     _, _, _, lca = ca_shape
# else:
#     raise ValueError(f"Unexpected Ca FITS shape: {ca_shape}")

# # ---- Si FITS ----
# with fits.open(SI_FITS, memmap=True) as f:
#     si_shape = f[0].data.shape

# if len(si_shape) == 5:
#     _, _, _, _, lsi = si_shape
# elif len(si_shape) == 4:
#     _, _, _, lsi = si_shape
# else:
#     raise ValueError(f"Unexpected Si FITS shape: {si_shape}")

n_stokes = len(STOKES_IDX)

# print(f"Atmosphere shape: t={t}, y={y}, x={x}, ltau={ltau}")
# print(f"Ca λ points: {lca}")
# print(f"Si λ points: {lsi}")


def get_valid_indices(stic_h5, wav_index=520, stokes_index=0, thr=3):
    with h5py.File(stic_h5, "r") as f:
        # read only one slice → cheap
        intensity = f["profiles"][0, :, :, wav_index, stokes_index]

    # find valid pixels
    y, x = np.where(intensity < thr)

    # build list of tuples
    idx = [(0, int(yy), int(xx)) for yy, xx in zip(y, x)]
    return idx


# ============================================================
# CACHE CHECK
# ============================================================
if os.path.exists(DATA_CACHE):

    Ca, Si, Y, train_idx, val_idx, test_idx = load_dataset_cache(DATA_CACHE)

    full_ds = CachedDataset(Ca, Si, Y)

else:

    print("No dataset cache found — building dataset...")

    valid_idx = get_valid_indices(STIC_h5)
    valid_idx.sort(key=lambda p: (p[0], p[1], p[2]))

    full_ds = CaSiAtmosDataset(STIC_h5, ATM_H5, valid_idx)

    N = len(full_ds)
    all_idx = np.arange(N)

    train_idx, val_idx, test_idx = make_splits(
        all_idx,
        TRAIN_SPLIT,
        VAL_SPLIT,
        seed=42
    )

    # --- save cache ---
    save_dataset_cache(
        DATA_CACHE,
        full_ds.Ca,
        full_ds.Si,
        full_ds.Y,
        train_idx,
        val_idx,
        test_idx
    )


# ============================================================
# FINAL DATASETS
# ============================================================
train_ds = Subset(full_ds, train_idx)
val_ds   = Subset(full_ds, val_idx)
test_ds  = Subset(full_ds, test_idx)


# # ============================================================
# # 4. SPLITS
# # ============================================================
# train_idx, val_idx, test_idx = make_splits(
#     valid_idx,
#     TRAIN_SPLIT,
#     VAL_SPLIT,
#     seed=42,
#     spatial_block=32   # optional but recommended
# )

# train_ds = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, train_idx)
# val_ds   = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, val_idx)
# test_ds = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, test_idx)

# train_ds = CaSiAtmosDataset(STIC_h5, ATM_H5, train_idx)
# val_ds   = CaSiAtmosDataset(STIC_h5, ATM_H5, val_idx)
# test_ds = CaSiAtmosDataset(STIC_h5, ATM_H5, test_idx)


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

stokes_scale = build_stokes_scale(
    n_stokes=n_stokes,
    stokes_indices=STOKES_IDX,
    scale_dict=STOKES_SCALE,
)

model = CaSiInversionCNN(
    n_stokes=n_stokes,
    ltau=ltau,
    ca_weight=ca_weight,
    si_weight=si_weight,
    stokes_scale=stokes_scale
)

# move to device first
model = model.to(device)

# wrap for multi-GPU if requested
if gpu_mode == "multi":
    model = torch.nn.DataParallel(model)

# unwrap model if using DataParallel
model_core = model.module if hasattr(model, "module") else model

print("Model loaded successfully")
print("Ca λ mask shape:", model_core.ca_encoder.w_lambda.shape)
print("Si λ mask shape:", model_core.si_encoder.w_lambda.shape)
print("Stokes scale:", model_core.ca_encoder.w_stokes.squeeze())

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
state = torch.load(CHECKPOINT_PATH, map_location=device)

if isinstance(model, torch.nn.DataParallel):
    model.module.load_state_dict(state)
else:
    model.load_state_dict(state)

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
