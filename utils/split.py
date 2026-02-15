import numpy as np


def make_splits(
    data,
    train_frac,
    val_frac,
    seed=0,
    spatial_block=None,
):
    """
    data can be:
        - tuple (t,y,x) shape
        - list[(t,y,x)]
        - array/list of integers
    """

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------
    # CASE 1: data is shape tuple → build full grid
    # ----------------------------------------------------
    if isinstance(data, tuple):
        t, y, x = data
        idx = [(i, j, k) for i in range(t)
                           for j in range(y)
                           for k in range(x)]
        idx = np.array(idx, dtype=np.int32)

    # ----------------------------------------------------
    # CASE 2: already index list or array
    # ----------------------------------------------------
    else:
        idx = np.array(data)

    # ----------------------------------------------------
    # OPTIONAL spatial split (only meaningful for tuples)
    # ----------------------------------------------------
    if spatial_block is not None and idx.ndim == 2 and idx.shape[1] == 3:

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

        return collect(train_tiles), collect(val_tiles), collect(test_tiles)

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

    # ----------------------------------------------------
    # RETURN TYPE FIX
    # ----------------------------------------------------
    # if idx is tuples → return tuples
    # if idx is integers → return integers
    if idx.ndim == 2:
        train_idx = [tuple(p) for p in train_idx]
        val_idx   = [tuple(p) for p in val_idx]
        test_idx  = [tuple(p) for p in test_idx]
    else:
        train_idx = train_idx.tolist()
        val_idx   = val_idx.tolist()
        test_idx  = test_idx.tolist()

    return train_idx, val_idx, test_idx
