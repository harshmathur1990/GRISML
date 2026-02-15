import numpy as np
import os


def save_dataset_cache(path, Ca, Si, Y, train_idx, val_idx, test_idx):
    print("Saving dataset cache:", path)
    np.savez_compressed(
        path,
        Ca=Ca,
        Si=Si,
        Y=Y,
        train_idx=np.array(train_idx),
        val_idx=np.array(val_idx),
        test_idx=np.array(test_idx),
    )


def load_dataset_cache(path):
    print("Loading dataset cache:", path)
    data = np.load(path, allow_pickle=True)

    Ca = data["Ca"]
    Si = data["Si"]
    Y  = data["Y"]

    train_idx = data["train_idx"].tolist()
    val_idx   = data["val_idx"].tolist()
    test_idx  = data["test_idx"].tolist()

    return Ca, Si, Y, train_idx, val_idx, test_idx
