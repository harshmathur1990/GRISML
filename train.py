import os
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["NCCL_SHM_DISABLE"] = "0"
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # optional, helps stability
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import h5py
from astropy.io import fits
import torch
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config import *
from data.datasets import CaSiAtmosDataset
# from data.bifrostdataset import CaSiAtmosDataset
# from data.cacheddataset import CachedDataset
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

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

    print(torch.cuda.get_device_capability())

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
    t, y, x, ltau = f["temp"].shape

n_stokes = len(STOKES_IDX)


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
# CHECK TARGET RANGE FROM DATALOADER
# ============================================================
def check_target_range(y, ranges, tag=""):
    with torch.no_grad():
        y_np = y.detach().cpu().numpy()

        ltau = y_np.shape[1] // 4

        temp  = y_np[:, :ltau]
        vlos  = y_np[:, ltau:2*ltau]
        vturb = y_np[:, 2*ltau:3*ltau]
        blong = y_np[:, 3*ltau:]

        checks = {
            "temp":  temp,
            "vlos":  vlos,
            "vturb": vturb,
            "blong": blong,
        }

        for name, arr in checks.items():
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            lo, hi = ranges[name]

            # small tolerance for float differences
            tol = 1e-5 * max(abs(lo), abs(hi), 1.0)

            if mn < lo - tol or mx > hi + tol:
                print(
                    f"\n❌ TARGET RANGE MISMATCH [{tag}] {name}\n"
                    f"batch = ({mn:.3e}, {mx:.3e})\n"
                    f"data  = ({lo:.3e}, {hi:.3e})"
                )


# ============================================================
# CHECK RANGE OF A DATALOADER (FAST)
# ============================================================
# ============================================================
# CHECK RANGE OF A DATALOADER + VERIFY AGAINST TARGET_RANGES
# ============================================================
def check_loader_ranges(loader, name, target_ranges=None, max_batches=50):

    print(f"\n=== CHECKING {name} LOADER RANGES ===")

    Ca_min, Ca_max = np.inf, -np.inf
    Si_min, Si_max = np.inf, -np.inf

    temp_min, temp_max   = np.inf, -np.inf
    vlos_min, vlos_max   = np.inf, -np.inf
    vturb_min, vturb_max = np.inf, -np.inf
    blong_min, blong_max = np.inf, -np.inf

    for i, (ca, si, y) in enumerate(loader):

        ca = ca.numpy()
        si = si.numpy()
        y  = y.numpy()

        Ca_min = min(Ca_min, np.nanmin(ca))
        Ca_max = max(Ca_max, np.nanmax(ca))
        Si_min = min(Si_min, np.nanmin(si))
        Si_max = max(Si_max, np.nanmax(si))

        ltau = y.shape[1] // 4

        temp  = y[:, :ltau]
        vlos  = y[:, ltau:2*ltau]
        vturb = y[:, 2*ltau:3*ltau]
        blong = y[:, 3*ltau:]

        temp_min  = min(temp_min,  np.nanmin(temp))
        temp_max  = max(temp_max,  np.nanmax(temp))
        vlos_min  = min(vlos_min,  np.nanmin(vlos))
        vlos_max  = max(vlos_max,  np.nanmax(vlos))
        vturb_min = min(vturb_min, np.nanmin(vturb))
        vturb_max = max(vturb_max, np.nanmax(vturb))
        blong_min = min(blong_min, np.nanmin(blong))
        blong_max = max(blong_max, np.nanmax(blong))

        if i + 1 >= max_batches:
            break

    print(f"Ca   range: {Ca_min:.3e} → {Ca_max:.3e}")
    print(f"Si   range: {Si_min:.3e} → {Si_max:.3e}")

    print("\n--- targets ---")
    print(f"Temp : {temp_min:.3e} → {temp_max:.3e}")
    print(f"Vlos : {vlos_min:.3e} → {vlos_max:.3e}")
    print(f"Vturb: {vturb_min:.3e} → {vturb_max:.3e}")
    print(f"Blong: {blong_min:.3e} → {blong_max:.3e}")

    # --------------------------------------------------------
    # VERIFY AGAINST EXPECTED TARGET RANGES
    # --------------------------------------------------------
    if target_ranges is not None:
        print("\n--- VERIFYING AGAINST TARGET_RANGES ---")

        observed = {
            "temp":  (temp_min, temp_max),
            "vlos":  (vlos_min, vlos_max),
            "vturb": (vturb_min, vturb_max),
            "blong": (blong_min, blong_max),
        }

        for name, (mn, mx) in observed.items():

            lo, hi = target_ranges[name]

            # tolerance:
            tol = 5e-3                    # float tolerance
            margin = 0.2 * (hi - lo)      # statistical tolerance

            bad = (
                mn < lo - tol - margin or
                mx > hi + tol + margin
            )

            if bad:
                print(
                    f"❌ RANGE VIOLATION {name}\n"
                    f"observed = ({mn:.3e}, {mx:.3e})\n"
                    f"expected = ({lo:.3e}, {hi:.3e})"
                )
            else:
                print(
                    f"✓ {name} within expected range"
                )


# ============================================================
# CACHE CHECK
# ============================================================
# if os.path.exists(DATA_CACHE):

#     Ca, Si, Y, train_idx, val_idx, test_idx = load_dataset_cache(DATA_CACHE)

#     full_ds = CachedDataset(Ca, Si, Y, logtemp=LOGTEMP)

# else:

#     print("No dataset cache found — building dataset...")

#     valid_idx = get_valid_indices(STIC_h5)
#     valid_idx.sort(key=lambda p: (p[0], p[1], p[2]))

#     full_ds = CaSiAtmosDataset(STIC_h5, ATM_H5, valid_idx, logtemp=LOGTEMP)

#     N = len(full_ds)
#     all_idx = np.arange(N)

#     train_idx, val_idx, test_idx = make_splits(
#         all_idx,
#         TRAIN_SPLIT,
#         VAL_SPLIT,
#         seed=42
#     )

#     # --- save cache ---
#     save_dataset_cache(
#         DATA_CACHE,
#         full_ds.Ca,
#         full_ds.Si,
#         full_ds.Y,
#         train_idx,
#         val_idx,
#         test_idx
#     )


# ============================================================
# DATASET RANGE CHECK
# ============================================================
# print("\n=== DATASET STATS ===")

# print("Ca  min/max:", np.nanmin(full_ds.Ca), np.nanmax(full_ds.Ca))
# print("Si  min/max:", np.nanmin(full_ds.Si), np.nanmax(full_ds.Si))
# print("Y   min/max:", np.nanmin(full_ds.Y),  np.nanmax(full_ds.Y))

# ---- per-parameter ranges ----
# ltau = full_ds.Y.shape[1] // 4

# temp  = full_ds.Y[:, :ltau]
# vlos  = full_ds.Y[:, ltau:2*ltau]
# vturb = full_ds.Y[:, 2*ltau:3*ltau]
# blong = full_ds.Y[:, 3*ltau:]

# print("\n--- PER VARIABLE ---")
# print("Temp  min/max:", np.nanmin(temp),  np.nanmax(temp))
# print("Vlos  min/max:", np.nanmin(vlos),  np.nanmax(vlos))
# print("Vturb min/max:", np.nanmin(vturb), np.nanmax(vturb))
# print("Blong min/max:", np.nanmin(blong), np.nanmax(blong))

# ============================================================
# TRAINING-TIME TARGET RANGE CHECK
# ============================================================
# print("\n=== TRAINING TARGET STATS ===")

# Ncheck = len(full_ds)
# idxs = np.linspace(0, len(full_ds)-1, Ncheck, dtype=int)

# print ("Ncheck: {}".format(Ncheck))
# y_all = []

# for i in idxs:
#     _, _, y = full_ds[i]
#     y_all.append(y.numpy())

# y_all = np.stack(y_all)

# print("Y(train) global min/max:",
#       np.nanmin(y_all),
#       np.nanmax(y_all))

# ltau = y_all.shape[1] // 4

# temp  = y_all[:, :ltau]
# vlos  = y_all[:, ltau:2*ltau]
# vturb = y_all[:, 2*ltau:3*ltau]
# blong = y_all[:, 3*ltau:]

# print("\n--- TRAINING RANGES ---")
# print("Temp  min/max:", np.nanmin(temp),  np.nanmax(temp))
# print("Vlos  min/max:", np.nanmin(vlos),  np.nanmax(vlos))
# print("Vturb min/max:", np.nanmin(vturb), np.nanmax(vturb))
# print("Blong min/max:", np.nanmin(blong), np.nanmax(blong))

# ============================================================
# STORE PHYSICAL TRAINING RANGES
# ============================================================
# ============================================================
# STORE TARGET RANGES USED FOR TRAINING
# ============================================================
# TARGET_RANGES = {
#     "temp":  (np.nanmin(temp),  np.nanmax(temp)),
#     "vlos":  (np.nanmin(vlos),  np.nanmax(vlos)),
#     "vturb": (np.nanmin(vturb), np.nanmax(vturb)),
#     "blong": (np.nanmin(blong), np.nanmax(blong)),
# }

# ============================================================
# FINAL DATASETS
# ============================================================
# train_ds = Subset(full_ds, train_idx)
# val_ds   = Subset(full_ds, val_idx)
# test_ds  = Subset(full_ds, test_idx)


# # ============================================================
# # 4. SPLITS
# # ============================================================
train_idx, val_idx, test_idx = make_splits(
    (t, y, x),
    TRAIN_SPLIT,
    VAL_SPLIT
)

train_ds = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, train_idx)
val_ds   = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, val_idx)
test_ds = CaSiAtmosDataset(CA_FITS, SI_FITS, ATM_H5, test_idx)

# train_ds = CaSiAtmosDataset(STIC_h5, ATM_H5, train_idx)
# val_ds   = CaSiAtmosDataset(STIC_h5, ATM_H5, val_idx)
# test_ds = CaSiAtmosDataset(STIC_h5, ATM_H5, test_idx)

# ============================================================
# 5. DATALOADERS (FINAL)
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


# check_loader_ranges(train_loader, "TRAIN", TARGET_RANGES, max_batches=200)
# check_loader_ranges(val_loader,   "VAL",   TARGET_RANGES, max_batches=200)
# check_loader_ranges(test_loader,  "TEST",  TARGET_RANGES, max_batches=200)

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

criterion = AtmosLoss(
    weights={
        "temp": 1.0,
        "vlos": 1.0,
        "blong": 1.0,
        "vturb": 1.0,
    },
    vturb_zero_weight=0.0   # forces vturb → 0 strongly
)

if DO_TRAIN:
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
        path=CHECKPOINT_PATH,
    )
else:
    print("\n🚀 DO_TRAIN=False → skipping training phase")

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


track_pred = TRACK_PRED

# ============================================================
# 9. TRAINING
# ============================================================
if DO_TRAIN:
    for ep in range(EPOCHS):

        # -------------------------
        # TRAIN
        # -------------------------
        model.train()
        train_loss = 0.0

        if track_pred:
            p_temp_min, p_temp_max   = np.inf, -np.inf
            p_vlos_min, p_vlos_max   = np.inf, -np.inf
            p_vturb_min, p_vturb_max = np.inf, -np.inf
            p_blong_min, p_blong_max = np.inf, -np.inf

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {ep:03d} [train]",
            leave=False,
        )

        for ca, si, y in train_pbar:

            # check_target_range(y, TARGET_RANGES, tag="train")

            ca = ca.to(device, non_blocking=True)
            si = si.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            pred = model(ca, si)
            loss = criterion(pred, y)

            # ------------------------------------------------------------
            # UPDATE PRED RANGE TRACKERS
            # ------------------------------------------------------------
            if track_pred:
                with torch.no_grad():
                    lt = pred.shape[1] // 4

                    p_temp_min  = min(p_temp_min,  pred[:, :lt].min().item())
                    p_temp_max  = max(p_temp_max,  pred[:, :lt].max().item())

                    p_vlos_min  = min(p_vlos_min,  pred[:, lt:2*lt].min().item())
                    p_vlos_max  = max(p_vlos_max,  pred[:, lt:2*lt].max().item())

                    p_vturb_min = min(p_vturb_min, pred[:, 2*lt:3*lt].min().item())
                    p_vturb_max = max(p_vturb_max, pred[:, 2*lt:3*lt].max().item())

                    p_blong_min = min(p_blong_min, pred[:, 3*lt:].min().item())
                    p_blong_max = max(p_blong_max, pred[:, 3*lt:].max().item())

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
                # check_target_range(y, TARGET_RANGES, tag="train")
                ca = ca.to(device, non_blocking=True)
                si = si.to(device, non_blocking=True)
                y  = y.to(device, non_blocking=True)

                pred = model(ca, si)
                loss = criterion(pred, y)

                if track_pred:
                    with torch.no_grad():
                        lt = pred.shape[1] // 4

                        p_temp_min  = min(p_temp_min,  pred[:, :lt].min().item())
                        p_temp_max  = max(p_temp_max,  pred[:, :lt].max().item())

                        p_vlos_min  = min(p_vlos_min,  pred[:, lt:2*lt].min().item())
                        p_vlos_max  = max(p_vlos_max,  pred[:, lt:2*lt].max().item())

                        p_vturb_min = min(p_vturb_min, pred[:, 2*lt:3*lt].min().item())
                        p_vturb_max = max(p_vturb_max, pred[:, 2*lt:3*lt].max().item())

                        p_blong_min = min(p_blong_min, pred[:, 3*lt:].min().item())
                        p_blong_max = max(p_blong_max, pred[:, 3*lt:].max().item())

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

        # ------------------------------------------------------------
        # PRINT PRED RANGE SUMMARY
        # ------------------------------------------------------------
        if track_pred:
            print("--- Prediction ranges (train epoch) ---")
            print(f"Temp : {p_temp_min:.3e} → {p_temp_max:.3e}")
            print(f"Vlos : {p_vlos_min:.3e} → {p_vlos_max:.3e}")
            print(f"Vturb: {p_vturb_min:.3e} → {p_vturb_max:.3e}")
            print(f"Blong: {p_blong_min:.3e} → {p_blong_max:.3e}")
            print("--- Prediction ranges (train epoch) ---\n")

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

if track_pred:
    p_temp_min, p_temp_max   = np.inf, -np.inf
    p_vlos_min, p_vlos_max   = np.inf, -np.inf
    p_vturb_min, p_vturb_max = np.inf, -np.inf
    p_blong_min, p_blong_max = np.inf, -np.inf

with torch.no_grad():
    for ca, si, y in tqdm(test_loader, desc="Test", leave=False):
        ca = ca.to(device, non_blocking=True)
        si = si.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        pred = model(ca, si)
        loss = criterion(pred, y)

        if track_pred:
            with torch.no_grad():
                lt = pred.shape[1] // 4

                p_temp_min  = min(p_temp_min,  pred[:, :lt].min().item())
                p_temp_max  = max(p_temp_max,  pred[:, :lt].max().item())

                p_vlos_min  = min(p_vlos_min,  pred[:, lt:2*lt].min().item())
                p_vlos_max  = max(p_vlos_max,  pred[:, lt:2*lt].max().item())

                p_vturb_min = min(p_vturb_min, pred[:, 2*lt:3*lt].min().item())
                p_vturb_max = max(p_vturb_max, pred[:, 2*lt:3*lt].max().item())

                p_blong_min = min(p_blong_min, pred[:, 3*lt:].min().item())
                p_blong_max = max(p_blong_max, pred[:, 3*lt:].max().item())

        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"FINAL TEST LOSS (unbiased): {test_loss:.3e}")

if track_pred:
    print("--- Prediction ranges (val epoch) ---")
    print(f"Temp : {p_temp_min:.3e} → {p_temp_max:.3e}")
    print(f"Vlos : {p_vlos_min:.3e} → {p_vlos_max:.3e}")
    print(f"Vturb: {p_vturb_min:.3e} → {p_vturb_max:.3e}")
    print(f"Blong: {p_blong_min:.3e} → {p_blong_max:.3e}")
    print("--- Prediction ranges (val epoch) ---\n")

