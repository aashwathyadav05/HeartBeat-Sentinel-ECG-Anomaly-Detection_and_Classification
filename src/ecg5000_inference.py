"""Shared inference service for the legacy ECG5000 autoencoder.

Kept separate from :mod:`src.unified_inference`. The 140-sample ECG5000
workflow must never be routed into the 12x1000 PTB-XL model.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from .model import ConvAutoencoder

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


class LegacyModelUnavailable(RuntimeError):
    """The ECG5000 checkpoint or threshold is missing/unusable."""


class ECG5000Inference:
    """Load the original ECG5000 CAE once and answer 140-sample queries."""

    def __init__(self, model_dir: Optional[str] = None, *, load_now: bool = True) -> None:
        self.model_dir = os.path.abspath(model_dir or DEFAULT_MODEL_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[ConvAutoencoder] = None
        self.threshold: Optional[float] = None
        self.seq_len = 140
        self.latent_dim = 32
        self.load_error: Optional[str] = None
        if load_now:
            self.load_artifacts()

    def load_artifacts(self) -> None:
        checkpoint_path = os.path.join(self.model_dir, "cae_best.pt")
        threshold_path = os.path.join(self.model_dir, "threshold.json")
        if not os.path.exists(checkpoint_path):
            raise LegacyModelUnavailable(
                f"ECG5000 checkpoint not found: {checkpoint_path}. Train it with 'python src/train.py'."
            )
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:  # older PyTorch
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as exc:
            raise LegacyModelUnavailable(f"Could not read ECG5000 checkpoint: {exc}") from exc
        self.seq_len = int(checkpoint.get("seq_len", 140))
        self.latent_dim = int(checkpoint.get("latent_dim", 32))
        model = ConvAutoencoder(seq_len=self.seq_len, latent_dim=self.latent_dim).to(self.device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise LegacyModelUnavailable(f"ECG5000 checkpoint is incompatible: {exc}") from exc
        model.eval()
        self.model = model
        if os.path.exists(threshold_path):
            with open(threshold_path, encoding="utf-8") as handle:
                info = json.load(handle)
            self.threshold = float(info["threshold"])
        else:
            LOGGER.warning("ECG5000 threshold file is missing; defaulting to 0.0")
            self.threshold = 0.0
        self.load_error = None

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.threshold is not None

    def health(self) -> Dict[str, Any]:
        return {
            "task": "ecg5000_anomaly",
            "available": self.is_available,
            "model_loaded": self.model is not None,
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "seq_len": int(self.seq_len),
            "error": self.load_error,
        }

    def predict(self, signal: Any) -> Dict[str, Any]:
        if self.model is None or self.threshold is None:
            raise LegacyModelUnavailable(
                self.load_error or "ECG5000 model is not loaded; train it with 'python src/train.py'."
            )
        array = np.asarray(signal, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != self.seq_len:
            raise ValueError(f"Signal must have exactly {self.seq_len} elements, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("Signal contains NaN or infinite values")
        tensor = torch.from_numpy(np.ascontiguousarray(array)).reshape(1, 1, -1).to(self.device)
        with torch.no_grad():
            reconstruction = self.model(tensor)
        original = tensor.squeeze().cpu().numpy()
        reconstructed = reconstruction.squeeze().cpu().numpy()
        pointwise = (original - reconstructed) ** 2
        segment_mse = float(pointwise.mean())
        predicted_label = "anomalous" if segment_mse > self.threshold else "normal"
        return {
            "segment_mse": segment_mse,
            "threshold": float(self.threshold),
            "predicted_label": predicted_label,
            "anomaly_score": segment_mse,
            "original": original.tolist(),
            "reconstructed": reconstructed.tolist(),
            "pointwise_squared_error": pointwise.tolist(),
            "explanation": (
                "Threshold decision made using validation-normal ECG5000 reconstruction "
                "errors at the saved threshold."
            ),
        }


def try_load_legacy(model_dir: Optional[str] = None):
    """Return ``(service, error)`` without raising at web startup."""
    try:
        service = ECG5000Inference(model_dir=model_dir, load_now=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        service = ECG5000Inference(model_dir=model_dir, load_now=False)
        service.load_error = str(exc)
        return service, str(exc)
    return service, None
