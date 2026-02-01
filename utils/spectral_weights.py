import torch

# ============================================================
# Helper: build wavelength weight mask
# ============================================================
def build_weight_mask(
    n_lambda: int,
    core_range=None,        # tuple (start, end)
    ignore_range=None,      # tuple (start, end)
    core_weight: float = 3.0,
    wing_weight: float = 1.0,
    ignore_weight: float = 0.0,
) -> torch.Tensor:
    """
    Returns a tensor of shape (n_lambda,) containing wavelength weights.
    """
    w = torch.full((n_lambda,), wing_weight, dtype=torch.float32)

    if core_range is not None:
        a, b = core_range
        w[a:b] = core_weight

    if ignore_range is not None:
        a, b = ignore_range
        w[a:b] = ignore_weight

    return w
