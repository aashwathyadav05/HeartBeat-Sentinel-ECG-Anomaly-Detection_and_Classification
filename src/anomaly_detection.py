import os
import json
import numpy as np
import torch

from .model import ConvAutoencoder
from .preprocess import prepare_dataloaders


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_artifacts(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(MODEL_DIR, "cae_best.pt"), map_location=device)
    seq_len = ckpt["seq_len"]
    latent_dim = ckpt.get("latent_dim", 32)
    model = ConvAutoencoder(seq_len=seq_len, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device, ckpt


def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
    # x, x_hat: (B,1,L)
    err = (x - x_hat) ** 2
    # segment-level MSE
    seg_err = err.mean(dim=(1, 2)).cpu().numpy()
    return seg_err


def pointwise_error(x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
    err = (x - x_hat) ** 2
    return err.squeeze(1).cpu().numpy()  # (B,L)


def get_errors(loader, model, device):
    all_seg = []
    all_pt = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            all_seg.append(reconstruction_error(xb, out))
            all_pt.append(pointwise_error(xb, out))
    return np.concatenate(all_seg), np.concatenate(all_pt, axis=0)


def select_threshold(method: str = "percentile", percentile: float = 95.0, factor: float = 1.5):
    model, device, _ = load_artifacts()
    info = prepare_dataloaders()
    val_loader = info["val_loader"]
    seg_err_val, _ = get_errors(val_loader, model, device)
    if method == "percentile":
        thresh = np.percentile(seg_err_val, percentile)
    elif method == "mean_std":
        thresh = seg_err_val.mean() + factor * seg_err_val.std()
    else:
        raise ValueError("Unknown method")
    res = {"threshold": float(thresh), "method": method, "percentile": percentile, "factor": factor, "val_seg_err_mean": float(seg_err_val.mean()), "val_seg_err_std": float(seg_err_val.std())}
    with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.save(os.path.join(DATA_OUT, "val_seg_errors.npy"), seg_err_val)
    return thresh
