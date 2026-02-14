import numpy as np


def make_splits(
    data,
    train_frac,
    val_frac,
    seed=0,
    spatial_block=None,
):
    """
    Flexible dataset splitter.

    Parameters
    ----------
    data : tuple OR list
        - (t,y,x) → build full grid indices
        - list[(t,y,x)] → split existing indices

    train_frac : float
    val_frac   : float
    seed       : int, reproducibility
    spatial_block : int or None
        If given, performs block-based split to avoid spatial leakage.
        Example: spatial_block=32 → splits by 32×32 tiles.

    Returns
    -------
    train_idx, val_idx, test_idx : list[(t,y,x)]
    """

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------
    # CASE 1: data is shape tuple → build full grid
    # ----------------------------------------------------
    if isinstance(data, tuple):
        t, y, x = data
        idx = np.array(
            [(i, j, k) for i in range(t)
                        for j in range(y)
                        for k in range(x)],
            dtype=np.int32
        )

    # ----------------------------------------------------
    # CASE 2: already index list → convert to array
    # ----------------------------------------------------
    else:
        idx = np.array(data, dtype=np.int32)

    # ----------------------------------------------------
    # OPTIONAL: spatial block split (recommended for CNNs)
    # ----------------------------------------------------
    if spatial_block is not None:

        # group pixels into spatial tiles
        tiles = {}

        for t_, y_, x_ in idx:
            key = (
                t_,
                y_ // spatial_block,
                x_ // spatial_block,
            )
            tiles.setdefault(key, []).append((t_, y_, x_))

        tile_keys = list(tiles.keys())
        rng.shuffle(tile_keys)

        n_tiles = len(tile_keys)
        n_tr  = int(train_frac * n_tiles)
        n_val = int(val_frac  * n_tiles)

        train_tiles = tile_keys[:n_tr]
        val_tiles   = tile_keys[n_tr:n_tr+n_val]
        test_tiles  = tile_keys[n_tr+n_val:]

        def collect(keys):
            return [pix for k in keys for pix in tiles[k]]

        train_idx = collect(train_tiles)
        val_idx   = collect(val_tiles)
        test_idx  = collect(test_tiles)

        return train_idx, val_idx, test_idx

    # ----------------------------------------------------
    # STANDARD RANDOM SPLIT
    # ----------------------------------------------------
    rng.shuffle(idx)

    n = len(idx)
    n_tr  = int(train_frac * n)
    n_val = int(val_frac  * n)

    train_idx = idx[:n_tr]
    val_idx   = idx[n_tr:n_tr+n_val]
    test_idx  = idx[n_tr+n_val:]

    # convert back to Python tuples (Dataset-friendly)
    train_idx = [tuple(p) for p in train_idx]
    val_idx   = [tuple(p) for p in val_idx]
    test_idx  = [tuple(p) for p in test_idx]

    return train_idx, val_idx, test_idx
