from pathlib import Path


base_path = Path('/mn/stornext/u3/harshm/Documents/Data/GRIS')
data_path = base_path / 'KMeans-Inversions' / 'fulldata_inversions'

# CA_FITS  = base_path / 'spectralveil_corrected_25Apr25ARM2-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
# SI_FITS  = base_path / 'spectralveil_corrected_25Apr25ARM1-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
# ATM_H5   = data_path / 'combined_output_004_atmos_B_cycle_1.nc'


STIC_h5 = base_path / 'bifrost_output.nc'
ATM_H5 = base_path / 'bifrost.nc'

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

# config.py

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
