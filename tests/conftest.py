"""Shared pytest fixtures.

All tests use synthetic metadata/signals. No real PTB-XL download, training
run or evaluation is required or performed by the unit test suite.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ptbxl_data import (  # noqa: E402
    LEAD_ORDER,
    TARGET_CATEGORIES,
    NormalizationStats,
    PreprocessingConfig,
)
from src.unified_artifacts import (  # noqa: E402
    default_threshold_payload,
    save_unified_artifacts,
)
from src.unified_ecg_model import (  # noqa: E402
    UnifiedECGModel,
    UnifiedModelConfig,
)


def make_ecg(length: int = 1000, seed: int = 7) -> np.ndarray:
    """Synthetic validated-shape ECG with finite values."""
    rng = np.random.RandomState(seed)
    return rng.randn(len(LEAD_ORDER), length).astype(np.float32)


def make_tiny_model(seed: int = 0) -> UnifiedECGModel:
    torch.manual_seed(seed)
    config = UnifiedModelConfig(
        sequence_length=1000,
        encoder_channels=(8, 16),
        decoder_channels=(8, 16),
        classifier_hidden=16,
        dropout=0.0,
    )
    return UnifiedECGModel(config)


@pytest.fixture
def artifact_factory(tmp_path):
    """Build complete versioned artifacts in a temporary directory.

    Returns a dict with the version directory, source model, preprocessing,
    normalization and threshold payload so tests can assert exact round-trips.
    """

    created: List[str] = []

    def _make(
        *,
        version: Optional[str] = None,
        anomaly_threshold: float = 0.5,
        class_thresholds: Optional[List[float]] = None,
        save_thresholds: bool = True,
        normalization_mode: str = "train_lead_zscore",
    ) -> Dict[str, Any]:
        version = version or f"v{len(created) + 1}"
        root = tmp_path / "artifacts" / "ptbxl_unified"
        version_dir = root / version
        model = make_tiny_model(seed=len(created) + 3)
        preprocessing = PreprocessingConfig(
            sampling_rate=100,
            sequence_length=1000,
            normalization=normalization_mode,
            clip_value=5.0,
        )
        normalization = NormalizationStats(
            mode=normalization_mode,
            mean=[0.25] * len(LEAD_ORDER),
            std=[2.0] * len(LEAD_ORDER),
            clip_value=5.0,
        )
        dataset_info = {
            "name": "PTB-XL",
            "version": "1.0.3",
            "sampling_rate": 100,
            "sequence_length": 1000,
            "lead_order": list(LEAD_ORDER),
            "fold_strategy": {"train": list(range(1, 9)), "val": 9, "test": 10},
            "label_order": list(TARGET_CATEGORIES),
            "stats": {
                "dataset_version": "1.0.3",
                "min_likelihood": 0.0,
                "n_records": 10,
                "n_usable_for_supervision": 8,
            },
        }
        thresholds = default_threshold_payload(
            anomaly_threshold,
            class_thresholds or [0.5] * len(TARGET_CATEGORIES),
        )
        paths = save_unified_artifacts(
            str(version_dir),
            model,
            preprocessing=preprocessing,
            normalization=normalization,
            dataset_info=dataset_info,
            training_config={"stage": "B", "seed": 42},
            thresholds=thresholds if save_thresholds else None,
            history={"train_classification_loss": [1.0, 0.5]},
        )
        created.append(version)
        return {
            "version_dir": paths.version_dir,
            "model": model,
            "preprocessing": preprocessing,
            "normalization": normalization,
            "thresholds": thresholds,
            "dataset_info": dataset_info,
        }

    return _make


@pytest.fixture
def synthetic_artifact(artifact_factory):
    return artifact_factory()
