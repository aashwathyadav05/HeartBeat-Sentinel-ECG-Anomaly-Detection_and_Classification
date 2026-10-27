import os
from typing import Tuple

import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


def load_ecg5000_txt(train_path: str = None, test_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECG5000 from space-delimited .txt files.
    First column is class label, remaining columns are waveform samples (length 140).
    Returns: (X_train, y_train, X_test, y_test)
    """
    if train_path is None:
        train_path = os.path.join(DATA_DIR, "ECG5000_TRAIN.txt")
    if test_path is None:
        test_path = os.path.join(DATA_DIR, "ECG5000_TEST.txt")

    # Load train
    df_train = pd.read_csv(train_path, sep=r'\s+', header=None, engine='python')
    df_test = pd.read_csv(test_path, sep=r'\s+', header=None, engine='python')

    y_train = df_train.iloc[:, 0].values.astype(float)
    X_train = df_train.iloc[:, 1:].values.astype(np.float32)

    y_test = df_test.iloc[:, 0].values.astype(float)
    X_test = df_test.iloc[:, 1:].values.astype(np.float32)

    return X_train, y_train, X_test, y_test


def load_ecg5000_any() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Try to load from .txt first (present in repo), fallback to arff/ts if needed."""
    train_txt = os.path.join(DATA_DIR, "ECG5000_TRAIN.txt")
    test_txt = os.path.join(DATA_DIR, "ECG5000_TEST.txt")
    if os.path.exists(train_txt) and os.path.exists(test_txt):
        return load_ecg5000_txt(train_txt, test_txt)
    raise FileNotFoundError(f"Could not find ECG5000 files in {DATA_DIR}. Expected ECG5000_TRAIN.txt and ECG5000_TEST.txt")
