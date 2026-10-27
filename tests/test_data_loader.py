import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_ecg5000_any


def test_load_shapes():
    Xtr, ytr, Xte, yte = load_ecg5000_any()
    assert Xtr.shape[1] == 140
    assert Xte.shape[1] == 140
    assert Xtr.shape[0] == len(ytr)
    assert Xte.shape[0] == len(yte)
    print("test_load_shapes passed")


if __name__ == "__main__":
    test_load_shapes()
