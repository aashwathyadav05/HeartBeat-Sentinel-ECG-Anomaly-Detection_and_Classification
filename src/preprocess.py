import os
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .data_loader import load_ecg5000_any


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
    normal_label: float = 1.0,
) -> Dict[str, Any]:
    set_seed(seed)
    X_train, y_train, X_test, y_test = load_ecg5000_any()

    # Select normal-labelled training segments for autoencoder training
    mask_normal_train = np.isclose(y_train, normal_label)
    X_train_normal = X_train[mask_normal_train]
    y_train_normal = y_train[mask_normal_train]

    # Split normal training data into train/val
    X_tr, X_val = train_test_split(X_train_normal, test_size=val_split, random_state=seed, shuffle=True)

    # Convert to tensors (B, 1, L)
    X_tr_t = torch.from_numpy(X_tr).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val).unsqueeze(1)
    X_test_t = torch.from_numpy(X_test).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_tr_t, X_tr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, X_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, X_test_t), batch_size=batch_size, shuffle=False)

    info = {
        "X_train_all": X_train,
        "y_train_all": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_tr": X_tr,
        "X_val": X_val,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "normal_label": normal_label,
        "seq_len": X_train.shape[1],
    }
    np.save(os.path.join(DATA_OUT, "test_data.npy"), X_test)
    np.save(os.path.join(DATA_OUT, "test_labels.npy"), y_test)
    np.save(os.path.join(DATA_OUT, "val_normal.npy"), X_val)
    np.save(os.path.join(DATA_OUT, "train_normal.npy"), X_tr)
    return info


def visualize_samples(n: int = 3, normal_label: float = 1.0):
    set_seed(123)
    X_train, y_train, X_test, y_test = load_ecg5000_any()
    classes = np.unique(y_train)
    plt.figure(figsize=(12, 4 * len(classes)))
    for i, c in enumerate(classes):
        mask = np.isclose(y_train, c)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        pick = np.random.choice(idxs, min(n, len(idxs)), replace=False)
        for j, pid in enumerate(pick):
            ax = plt.subplot(len(classes), n, i * n + j + 1)
            ax.plot(X_train[pid])
            ax.set_title(f"Train class {c}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "train_samples_by_class.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")
    return out
