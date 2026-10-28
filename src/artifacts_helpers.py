"""Internal helpers used by the staged unified-model trainer."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Mapping, Optional

from .ptbxl_data import (
    PTBXLBundle,
    NormalizationStats,
    PreprocessingConfig,
    artifact_dataset_info,
)
from .unified_artifacts import (
    artifact_paths,
    save_unified_artifacts,
)
from .unified_ecg_model import UnifiedECGModel

LOGGER = logging.getLogger(__name__)

CURRENT_TRAINING_NOTES = [
    "Folds 1-8 are used for fitting; fold 9 for early stopping and threshold tuning; fold 10 is test-only.",
    "Unknown labels are masked out and are never converted to negative targets.",
    "Class imbalance uses pos_weight computed from the training partition only.",
    "Stage C restricts reconstruction loss to NORM-only records via an explicit batch mask.",
    "Sigmoid classifier outputs are not calibrated clinical probabilities.",
]


def build_stage_metadata(stage: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Small JSON-safe metadata block describing the training run."""
    return {
        "training_stage": stage,
        "notes": CURRENT_TRAINING_NOTES,
        "run_settings": {
            "sampling_rate": int(getattr(args, "sampling_rate", 100)),
            "seed": int(getattr(args, "seed", 42)),
            "batch_size": int(getattr(args, "batch_size", 16)),
            "max_train_records": getattr(args, "max_train_records", None),
            "max_val_records": getattr(args, "max_val_records", None),
        },
    }


def save_final_artifacts(
    out_dir: str,
    model: UnifiedECGModel,
    *,
    preprocessing: PreprocessingConfig,
    normalization: NormalizationStats,
    bundle: PTBXLBundle,
    training_config: Mapping[str, Any],
    history: Optional[Mapping[str, Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Save metadata/checkpoint/history; thresholds are tuned separately.

    Any pre-existing ``thresholds.json`` is left in place so that ``skip-tune``
    does not silently replace frozen thresholds. A warning makes stale
    thresholds visible; a fresh artifact without tuning lacks thresholds and
    inference reports an incomplete artifact.
    """
    paths = artifact_paths(out_dir)
    if os.path.exists(paths.thresholds_path):
        LOGGER.warning(
            "Existing thresholds remain at %s. Run the 'tune' command if this model was retrained.",
            paths.thresholds_path,
        )
    save_unified_artifacts(
        out_dir,
        model,
        preprocessing=preprocessing,
        normalization=normalization,
        dataset_info=artifact_dataset_info(bundle, preprocessing),
        training_config=dict(training_config),
        thresholds=None,
        history=dict(history) if history is not None else None,
        extra_metadata=extra_metadata,
    )
    return paths.version_dir
