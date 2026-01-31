import numpy as np


def make_splits(shape, train_frac, val_frac):
    t, y, x = shape
    idx = [(i,j,k) for i in range(t)
                    for j in range(y)
                    for k in range(x)]
    np.random.shuffle(idx)

    n = len(idx)
    n_tr = int(train_frac * n)
    n_val = int(val_frac * n)

    return idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]
