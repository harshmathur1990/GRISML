import torch

def build_weight_mask(n_lambda, core_range=None, ignore_range=None,
                      core_weight=3.0, wing_weight=1.0, ignore_weight=0.0):
    """
    Returns (n_lambda,) tensor
    """
    w = torch.full((n_lambda,), wing_weight)

    if core_range is not None:
        w[core_range[0]:core_range[1]] = core_weight

    if ignore_range is not None:
        w[ignore_range[0]:ignore_range[1]] = ignore_weight

    return w
