from pathlib import Path


base_path = Path('/mn/stornext/u3/harshm/Documents/Data/GRIS')
data_path = base_path / 'KMeans-Inversions' / 'fulldata_inversions'

CA_FITS  = base_path / 'spectralveil_corrected_25Apr25ARM2-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
SI_FITS  = base_path / 'spectralveil_corrected_25Apr25ARM1-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
ATM_H5   = data_path / 'combined_output_atmos_cycle_B_3.nc'

BATCH_SIZE = 32
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