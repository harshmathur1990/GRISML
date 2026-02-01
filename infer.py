import h5py
import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm

from config import *
from models.model import CaSiInversionCNN
from utils.denormalise import denormalise_output
from utils.spectral_weights import build_weight_mask
from utils.stokes_scaling import build_stokes_scale


def get_ltau_scale():
    taumin = -7.8
    taumax= 1.0
    dtau = 0.14
    ntau = int((taumax-taumin)/dtau) + 1
    ltau_scale = np.arange(ntau, dtype='float64')/(ntau-1.0) * (taumax-taumin) + taumin

    return ltau_scale


# =========================
#   LOAD INPUT DATA
# =========================
def load_fits_data(fname):
    """
    Returns data and a flag indicating if time dimension exists
    """
    with fits.open(fname, memmap=True) as f:
        data = f[0].data

    if data.ndim == 5:
        # (t, s, y, x, λ)
        return data, True
    elif data.ndim == 4:
        # (s, y, x, λ) → add fake time dim
        return data[None, ...], False
    else:
        raise ValueError(f"Unexpected FITS shape: {data.shape}")


# =========================
#        MAIN
# =========================
def run_inference(ca_fits, si_fits, atm_out_h5, batch_size=1024):

    # ---- load inputs ----
    ca, has_time = load_fits_data(ca_fits)
    si, _ = load_fits_data(si_fits)

    t, s, y, x, _ = ca.shape
    n_stokes = len(STOKES_IDX)

    # ---- load ltau size from model checkpoint ----
    with h5py.File(ATM_H5, "r") as f:
        ltau = f["temp"].shape[-1]

    # ---- build wavelength masks (SAME AS TRAINING) ----
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

    # ---- model ----
    device = torch.device(DEVICE)

    model = CaSiInversionCNN(
        n_stokes=n_stokes,
        ltau=ltau,
        ca_weight=ca_weight,
        si_weight=si_weight,
        stokes_scale=stokes_scale
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # =========================
    #   MODEL SANITY CHECK
    # =========================
    print("Model loaded successfully")
    print("Ca λ mask shape:", model.ca_encoder.w_lambda.shape)
    print("Si λ mask shape:", model.si_encoder.w_lambda.shape)
    print("Stokes scale:", model.ca_encoder.w_stokes.squeeze())

    # ---- output arrays (NORMALISED) ----
    out = np.zeros((t, y, x, 4 * ltau), dtype=np.float32)

    # =========================
    #    PIXEL-WISE INFERENCE
    # =========================
    pixels = [(ti, yi, xi)
              for ti in range(t)
              for yi in range(y)
              for xi in range(x)]

    for i in tqdm(
        range(0, len(pixels), batch_size),
        desc="Inference",
    ):
        batch = pixels[i:i+batch_size]

        ca_batch = []
        si_batch = []

        for ti, yi, xi in batch:
            ca_batch.append(
                ca[ti, STOKES_IDX, yi, xi, :]
            )
            si_batch.append(
                si[ti, STOKES_IDX, yi, xi, :]
            )

        ca_batch = torch.tensor(
            np.stack(ca_batch), dtype=torch.float32
        ).to(device)

        si_batch = torch.tensor(
            np.stack(si_batch), dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            pred = model(ca_batch, si_batch)

        pred = pred.cpu().numpy()

        for j, (ti, yi, xi) in enumerate(batch):
            out[ti, yi, xi, :] = pred[j]


    # =========================
    #   DE-NORMALISE & SAVE
    # =========================
    atm = denormalise_output(out.reshape(-1, 4 * ltau), ltau)

    with h5py.File(atm_out_h5, "w") as f:

        ltau_scale = get_ltau_scale()

        shape = (t, y, x, ltau)

        ltau500 = np.broadcast_to(ltau_scale[None, None, None, :], shape)

        f.create_dataset("temp",  data=atm["temp"].reshape(shape))
        f.create_dataset("vlos",  data=atm["vlos"].reshape(shape))
        f.create_dataset("vturb", data=atm["vturb"].reshape(shape))
        f.create_dataset("blong", data=atm["blong"].reshape(shape))
        f.create_dataset("ltau500", data=ltau500)

    print(f"Saved output to {atm_out_h5}")

    # ---- if input had no time dim, squeeze it ----
    if not has_time:
        with h5py.File(atm_out_h5, "r+") as f:
            for k in f.keys():
                f[k][...] = f[k][0]


# =========================
#        CLI ENTRY
# =========================
if __name__ == "__main__":
    import sys

    ca_fits = sys.argv[1]
    si_fits = sys.argv[2]
    out_h5  = sys.argv[3]

    run_inference(ca_fits, si_fits, out_h5)
