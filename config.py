from pathlib import Path


base_path = Path('/mn/stornext/u3/harshm/Documents/Data/GRIS')
data_path = base_path / 'KMeans-Inversions' / 'fulldata_inversions'
dev_shm = Path("/dev/shm")

# CA_FITS  = base_path / 'spectralveil_corrected_25Apr25ARM2-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
# SI_FITS  = base_path / 'spectralveil_corrected_25Apr25ARM1-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
# ATM_H5   = data_path / 'combined_output_004_atmos_B_cycle_1.nc'


STIC_h5 = dev_shm / 'bifrost_output.nc'
ATM_H5 = dev_shm / 'bifrost.nc'

DATA_CACHE = data_path / "dataset_cache.npz"

# options:
#   "stream"  → read pixels from HDF5 (best for cvfs)
#   "full"    → load cubes fully to RAM first (best if on /dev/shm)
DATA_LOADING_MODE = "stream"

BATCH_SIZE = 1024
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

OUTPUT_SCALES = {
    "temp": 1e-3,
    "vlos": 1e-5,
    "vturb": 1e-5,
    "blong": 1e-4,
}


LOSS_WEIGHTS = {
    "temp":  5.0,   # dominant physics
    "vlos":  1.5,   # medium
    "blong": 1.5,   # medium
    "vturb": 0.2,   # weak (almost ignored)
}

LOGTEMP = True

OUTPUT_MULTIPLIERS = {
    "vlos": 1e-1,     # shrink km/s range
    "vturb": 1e-1,
    "blong": 1e1,   # magnetic fields often huge
}

APPLY_OUTPUT_RESCALE = True

# config.py

# ============================================================
# RUN CONTROL
# ============================================================
DO_TRAIN = False     # False → skip training, only load best model + validate/test
TRACK_PRED = True

EARLY_STOPPING_PATIENCE = 10
CHECKPOINT_PATH = "best_model.pt"
MIN_DELTA = 0.0   # minimum improvement to count

# Stokes indices:
# 0 = I, 1 = Q, 2 = U, 3 = V
STOKES_IDX = [0, 3]

CA_N_WAVELENGTH = 1000
CA_CORE_RANGE = (0, 226)

# Si I
SI_N_WAVELENGTH = 872
SI_CORE_RANGE = (400, 656)
SI_IGNORE_RANGE = (656, 872)

# Importance weights
CORE_WEIGHT = 3.0      # boost core
WING_WEIGHT = 1.0      # normal
IGNORE_WEIGHT = 0.0    # hard ignore

STOKES_SCALE = {
    0: 1.0,
    1: 100.0,
    2: 100.0,
    3: 100.0,
}
