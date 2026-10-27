import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from .anomaly_detection import load_artifacts
from .preprocess import prepare_dataloaders


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def visualize_example(idx: int = 0, save: bool = True):
    model, device, _ = load_artifacts()
    info = prepare_dataloaders()
    X_test = info["X_test"]
    y_test = info["y_test"]

    x = torch.from_numpy(X_test[idx:idx + 1]).unsqueeze(1).to(device)
    with torch.no_grad():
        x_hat = model(x)
    x_np = x.squeeze().cpu().numpy()
    xh_np = x_hat.squeeze().cpu().numpy()
    pt_err = (x_np - xh_np) ** 2
    seg_err = pt_err.mean()

    try:
        with open(os.path.join(MODEL_DIR, "threshold.json")) as f:
            thresh = json.load(f)["threshold"]
    except Exception:
        thresh = seg_err

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x_np, label="Original")
    axes[0].plot(xh_np, label="Reconstructed")
    axes[0].set_title(f"ECG Segment {idx} | True label={y_test[idx]} | SegMSE={seg_err:.6f} | Thresh={thresh:.6f} | Flag={'ANOM' if seg_err>thresh else 'NORMAL'}")
    axes[0].legend()
    axes[1].plot(pt_err)
    axes[1].set_title("Point-wise Squared Reconstruction Error")
    axes[1].set_ylabel("Error")
    axes[2].axhline(0, color='k', linewidth=0.5)
    axes[2].plot(pt_err)
    axes[2].fill_between(range(len(pt_err)), pt_err, alpha=0.3)
    axes[2].set_title("Error Highlight")
    axes[2].set_xlabel("Time")
    plt.tight_layout()
    if save:
        out = os.path.join(MODEL_DIR, f"example_{idx}.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")
        return out
    plt.show()
    return None


if __name__ == "__main__":
    visualize_example(0)
