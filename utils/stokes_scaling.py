import torch


def build_stokes_scale(n_stokes, stokes_indices, scale_dict):
    """
    Returns a tensor of shape (n_stokes, 1) to scale Stokes channels.

    stokes_indices: e.g. [0, 3]
    scale_dict: dict mapping original Stokes index -> scale
    """
    scale = torch.ones(n_stokes, dtype=torch.float32)

    for i, s in enumerate(stokes_indices):
        if s in scale_dict:
            scale[i] = scale_dict[s]

    return scale[:, None]  # (stokes, 1)
