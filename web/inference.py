import os
import json
from typing import Dict, Any

import numpy as np
import torch

from src.model import ConvAutoencoder


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


class ECGInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.threshold = None
        self.seq_len = 140
        self.load_artifacts()

    def load_artifacts(self):
        ckpt_path = os.path.join(MODEL_DIR, "cae_best.pt")
        thresh_path = os.path.join(MODEL_DIR, "threshold.json")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.seq_len = ckpt.get("seq_len", 140)
        latent_dim = ckpt.get("latent_dim", 32)
        self.model = ConvAutoencoder(seq_len=self.seq_len, latent_dim=latent_dim).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        if os.path.exists(thresh_path):
            with open(thresh_path) as f:
                tinfo = json.load(f)
            self.threshold = float(tinfo["threshold"])
        else:
            self.threshold = 0.0

    def predict(self, signal: np.ndarray) -> Dict[str, Any]:
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        signal = signal.astype(np.float32).reshape(1, 1, -1)
        if signal.shape[2] != self.seq_len:
            raise ValueError(f"Signal length must be {self.seq_len}, got {signal.shape[2]}")
        x = torch.from_numpy(signal).to(self.device)
        with torch.no_grad():
            x_hat = self.model(x)
        x_np = x.squeeze().cpu().numpy()
        xh_np = x_hat.squeeze().cpu().numpy()
        pt_err = (x_np - xh_np) ** 2
        seg_err = float(pt_err.mean())
        pred_label = "anomalous" if seg_err > self.threshold else "normal"
        return {
            "segment_mse": seg_err,
            "threshold": float(self.threshold),
            "predicted_label": pred_label,
            "anomaly_score": seg_err,
            "original": x_np.tolist(),
            "reconstructed": xh_np.tolist(),
            "pointwise_squared_error": pt_err.tolist(),
            "explanation": "Threshold decision made using validation-normal reconstruction errors (class 1) at 95th percentile baseline."
        }


inference_engine = ECGInference()
