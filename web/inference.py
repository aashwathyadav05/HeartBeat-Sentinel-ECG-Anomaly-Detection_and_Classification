"""Backward-compatible ECG5000 inference import.

The implementation now lives in :mod:`src.ecg5000_inference` so Streamlit and
Flask share one service. Existing callers can keep importing
``web.inference.inference_engine``.
"""

from __future__ import annotations

from src.ecg5000_inference import (
    DEFAULT_MODEL_DIR,
    ECG5000Inference,
    LegacyModelUnavailable,
    try_load_legacy,
)

inference_engine, inference_error = try_load_legacy()

__all__ = [
    "ECG5000Inference",
    "LegacyModelUnavailable",
    "inference_engine",
    "inference_error",
    "DEFAULT_MODEL_DIR",
]
