# utils/denormalise.py

import numpy as np
from config import OUTPUT_SCALES


def denormalise_output(y_pred, ltau):
    """
    y_pred: (N, 4*ltau) tensor or numpy array
    returns dict with physical units
    """
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    s = OUTPUT_SCALES

    out = {}
    out["temp"]  = y_pred[:, 0*ltau:1*ltau] / s["temp"]
    out["vlos"]  = y_pred[:, 1*ltau:2*ltau] / s["vlos"]
    out["vturb"] = y_pred[:, 2*ltau:3*ltau] / s["vturb"]
    out["blong"] = y_pred[:, 3*ltau:4*ltau] / s["blong"]

    return out
