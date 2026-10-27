import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import ConvAutoencoder


def test_model_shape():
    model = ConvAutoencoder(seq_len=140)
    x = torch.randn(4, 1, 140)
    y = model(x)
    assert y.shape == x.shape, f"Shape mismatch {y.shape} vs {x.shape}"
    print("test_model_shape passed")


if __name__ == "__main__":
    test_model_shape()
