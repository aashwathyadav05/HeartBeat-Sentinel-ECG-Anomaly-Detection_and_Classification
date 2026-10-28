"""Versioned artifact handling for the unified PTB-XL model.

An artifact version directory contains:

* ``model.pt``          - state dict, model config and artifact format version
* ``metadata.json``     - label order, preprocessing/normalization, dataset and training config
* ``thresholds.json``   - frozen anomaly threshold and five class thresholds
* ``training_history.json`` (when training produced one)
* optional evaluation plots / JSON

The loader validates compatibility before building the model so an old or
mismatched checkpoint cannot silently produce predictions.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ptbxl_data import (
    LEAD_ORDER,
    PTBXL_DATASET_VERSION,
    NormalizationStats,
    PreprocessingConfig,
    TARGET_CATEGORIES,
)
from .unified_ecg_model import (
    MODEL_ARCHITECTURE,
    UnifiedECGModel,
    UnifiedModelConfig,
)

ARTIFACT_FORMAT_VERSION = 1
MODEL_VERSION = "1.0.0"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ARTIFACT_ROOT = os.path.join(PROJECT_ROOT, "artifacts", "ptbxl_unified")


class ArtifactError(RuntimeError):
    """Base class for artifact problems."""


class ArtifactNotFoundError(ArtifactError):
    """The requested artifact directory/checkpoint does not exist."""


class ArtifactIncompatibleError(ArtifactError):
    """The artifact exists but is incompatible with this codebase."""


@dataclass
class ArtifactPaths:
    version_dir: str
    model_path: str
    metadata_path: str
    thresholds_path: str
    history_path: str


def artifact_paths(version_dir: str) -> ArtifactPaths:
    version_dir = os.path.abspath(version_dir)
    return ArtifactPaths(
        version_dir=version_dir,
        model_path=os.path.join(version_dir, "model.pt"),
        metadata_path=os.path.join(version_dir, "metadata.json"),
        thresholds_path=os.path.join(version_dir, "thresholds.json"),
        history_path=os.path.join(version_dir, "training_history.json"),
    )


def _version_sort_key(name: str) -> Tuple[int, str]:
    match = re.search(r"v(\d+)", name)
    return (int(match.group(1)) if match else -1, name)


def resolve_artifact_dir(path: Optional[str] = None) -> str:
    """Resolve a version directory, model.pt path, latest pointer or root."""
    candidate = path or os.environ.get("HEARTBEAT_UNIFIED_ARTIFACTS") or DEFAULT_ARTIFACT_ROOT
    candidate = os.path.abspath(os.path.expanduser(str(candidate)))
    if os.path.isfile(candidate):
        if os.path.basename(candidate) == "latest.json":
            with open(candidate, encoding="utf-8") as handle:
                pointer = json.load(handle)
            target = pointer.get("version_dir") or pointer.get("path")
            if not target:
                raise ArtifactNotFoundError(f"Malformed latest pointer: {candidate}")
            candidate = target if os.path.isabs(target) else os.path.join(os.path.dirname(candidate), target)
            candidate = os.path.abspath(candidate)
        else:
            candidate = os.path.dirname(candidate)
    if os.path.isdir(candidate):
        if os.path.isfile(os.path.join(candidate, "model.pt")):
            return candidate
        latest = os.path.join(candidate, "latest.json")
        if os.path.isfile(latest):
            return resolve_artifact_dir(latest)
        versions = [
            os.path.join(candidate, name)
            for name in os.listdir(candidate)
            if os.path.isdir(os.path.join(candidate, name))
            and os.path.isfile(os.path.join(candidate, name, "model.pt"))
        ]
        if versions:
            return sorted(versions, key=lambda value: _version_sort_key(os.path.basename(value)))[-1]
    raise ArtifactNotFoundError(
        f"No unified PTB-XL artifacts found at '{candidate}'. Train the model first, for example:\n"
        "  python -m src.train_unified_model all --data-dir data/ptbxl\n"
        "or pass an explicit artifact directory with --artifacts / HEARTBEAT_UNIFIED_ARTIFACTS."
    )


def _write_json_atomic(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_latest_pointer(version_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(version_dir))
    pointer = os.path.join(parent, "latest.json")
    _write_json_atomic(pointer, {"version_dir": os.path.basename(os.path.abspath(version_dir))})
    return pointer


def save_unified_artifacts(
    version_dir: str,
    model: UnifiedECGModel,
    *,
    preprocessing: PreprocessingConfig,
    normalization: NormalizationStats,
    dataset_info: Mapping[str, Any],
    training_config: Mapping[str, Any],
    thresholds: Optional[Mapping[str, Any]] = None,
    history: Optional[Mapping[str, Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> ArtifactPaths:
    """Persist checkpoint, metadata and thresholds as one versioned artifact."""
    paths = artifact_paths(version_dir)
    os.makedirs(paths.version_dir, exist_ok=True)
    model_config = model.config.to_dict()
    checkpoint = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "architecture": MODEL_ARCHITECTURE,
        "model_config": model_config,
        "state_dict": model.state_dict(),
    }
    temporary = paths.model_path + ".tmp"
    import torch

    torch.save(checkpoint, temporary)
    os.replace(temporary, paths.model_path)

    metadata: Dict[str, Any] = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "architecture": MODEL_ARCHITECTURE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_order": list(TARGET_CATEGORIES),
        "model_config": model_config,
        "preprocessing": {
            **preprocessing.to_dict(),
            "normalization": normalization.to_dict(),
            "normalization_fit_source": (
                "training-partition normal (NORM-only) records only"
                if normalization.mode == "train_lead_zscore"
                else normalization.mode
            ),
            "input_units": "PTB-XL physical signal (mV) before training-set normalization",
            "bad_input_policy": "reject non-finite, wrong-shape and wrong-rate inputs; no implicit resampling",
        },
        "dataset": dict(dataset_info),
        "training": dict(training_config),
        "task_definitions": {
            "anomaly": "mean squared reconstruction error in normalized units vs frozen threshold",
            "classification": "five independent sigmoid scores; categories may coexist",
            "separation": (
                "Anomaly output is not a diagnosis. A classification score is not a calibrated "
                "clinical probability and does not establish a diagnosis."
            ),
        },
        "disclaimer": (
            "Educational research prototype. Not a medical device and not for clinical "
            "diagnosis or treatment decisions."
        ),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    _write_json_atomic(paths.metadata_path, metadata)

    if thresholds is not None:
        write_thresholds(version_dir, thresholds)
    if history is not None:
        _write_json_atomic(paths.history_path, dict(history))
    write_latest_pointer(paths.version_dir)
    return paths


def write_thresholds(version_dir: str, thresholds: Mapping[str, Any]) -> ArtifactPaths:
    """Validate and write ``thresholds.json`` atomically."""
    paths = artifact_paths(version_dir)
    payload = dict(thresholds)
    payload.setdefault("label_order", list(TARGET_CATEGORIES))
    _validate_threshold_payload(payload)
    _write_json_atomic(paths.thresholds_path, payload)
    return paths


def _validate_threshold_payload(payload: Mapping[str, Any]) -> None:
    if list(payload.get("label_order", [])) != list(TARGET_CATEGORIES):
        raise ArtifactIncompatibleError(
            f"Threshold label order must be {list(TARGET_CATEGORIES)}, got {payload.get('label_order')}"
        )
    anomaly = payload.get("anomaly_threshold")
    if anomaly is None or not np.isfinite(float(anomaly)) or float(anomaly) < 0:
        raise ArtifactIncompatibleError("thresholds.json needs a finite non-negative anomaly_threshold")
    class_thresholds = payload.get("class_thresholds")
    if not isinstance(class_thresholds, (list, tuple)) or len(class_thresholds) != len(TARGET_CATEGORIES):
        raise ArtifactIncompatibleError("thresholds.json needs five class_thresholds")
    for value in class_thresholds:
        if value is None or not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ArtifactIncompatibleError("class thresholds must be finite values in [0, 1]")


@dataclass
class LoadedUnifiedArtifacts:
    model: UnifiedECGModel
    model_config: UnifiedModelConfig
    preprocessing: PreprocessingConfig
    normalization: NormalizationStats
    metadata: Dict[str, Any]
    thresholds: Dict[str, Any]
    paths: ArtifactPaths


def read_artifact_summary(path: Optional[str] = None) -> Dict[str, Any]:
    """Cheap health/metadata read without instantiating the model."""
    version_dir = resolve_artifact_dir(path)
    paths = artifact_paths(version_dir)
    for required in (paths.model_path, paths.metadata_path, paths.thresholds_path):
        if not os.path.isfile(required):
            raise ArtifactNotFoundError(f"Artifact is incomplete; missing {required}")
    with open(paths.metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(paths.thresholds_path, encoding="utf-8") as handle:
        thresholds = json.load(handle)
    return {
        "version_dir": version_dir,
        "model_version": metadata.get("model_version"),
        "dataset": metadata.get("dataset", {}),
        "label_order": metadata.get("label_order"),
        "anomaly_threshold": thresholds.get("anomaly_threshold"),
        "class_thresholds": thresholds.get("class_thresholds"),
        "trained_utc": metadata.get("created_utc"),
    }


def load_unified_artifacts(
    path: Optional[str] = None,
    *,
    device: Optional[Any] = None,
) -> LoadedUnifiedArtifacts:
    """Load and validate a unified model artifact bundle."""
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version_dir = resolve_artifact_dir(path)
    paths = artifact_paths(version_dir)
    for required in (paths.model_path, paths.metadata_path, paths.thresholds_path):
        if not os.path.isfile(required):
            raise ArtifactNotFoundError(f"Artifact is incomplete; missing {required}")

    try:
        checkpoint = torch.load(paths.model_path, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch without weights_only
        checkpoint = torch.load(paths.model_path, map_location="cpu")
    except Exception as exc:
        raise ArtifactIncompatibleError(f"Could not read checkpoint {paths.model_path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ArtifactIncompatibleError("model.pt does not contain a checkpoint dictionary")
    format_version = checkpoint.get("artifact_format_version")
    if format_version != ARTIFACT_FORMAT_VERSION:
        raise ArtifactIncompatibleError(
            f"Checkpoint artifact format {format_version} is incompatible with supported "
            f"format {ARTIFACT_FORMAT_VERSION}. Retrain the unified model."
        )
    if checkpoint.get("architecture") != MODEL_ARCHITECTURE:
        raise ArtifactIncompatibleError(
            f"Checkpoint architecture {checkpoint.get('architecture')!r} is not {MODEL_ARCHITECTURE!r}"
        )
    if "state_dict" not in checkpoint:
        raise ArtifactIncompatibleError("Checkpoint is missing state_dict")

    try:
        with open(paths.metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        with open(paths.thresholds_path, encoding="utf-8") as handle:
            thresholds = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactIncompatibleError(f"Artifact JSON is malformed: {exc}") from exc

    metadata_format = metadata.get("artifact_format_version")
    if metadata_format != ARTIFACT_FORMAT_VERSION:
        raise ArtifactIncompatibleError(
            f"metadata.json artifact format {metadata_format} is incompatible with "
            f"supported format {ARTIFACT_FORMAT_VERSION}"
        )
    if list(metadata.get("label_order", [])) != list(TARGET_CATEGORIES):
        raise ArtifactIncompatibleError(
            f"Metadata label order must be {list(TARGET_CATEGORIES)}, got {metadata.get('label_order')}"
        )
    try:
        _validate_threshold_payload(thresholds)
    except ArtifactIncompatibleError:
        raise
    except Exception as exc:
        raise ArtifactIncompatibleError(f"Invalid thresholds.json: {exc}") from exc

    preprocessing_data = metadata.get("preprocessing")
    if not isinstance(preprocessing_data, dict):
        raise ArtifactIncompatibleError("metadata.json is missing the preprocessing configuration")
    try:
        # ``preprocessing.normalization`` is saved as the nested statistics
        # object. Older/hand-written payloads may store only the mode string,
        # so accept both shapes without feeding a dict to the mode validator.
        config_payload = dict(preprocessing_data)
        normalization_payload = config_payload.pop("normalization", None)
        if isinstance(normalization_payload, str):
            config_payload["normalization"] = normalization_payload
            normalization_payload = None
        preprocessing = PreprocessingConfig.from_dict(config_payload)
        if normalization_payload is None:
            normalization = NormalizationStats(
                mode=preprocessing.normalization,
                clip_value=preprocessing.clip_value,
            )
        else:
            normalization = NormalizationStats.from_dict(normalization_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactIncompatibleError(f"Invalid preprocessing configuration: {exc}") from exc
    if list(preprocessing.lead_order) != list(LEAD_ORDER):
        raise ArtifactIncompatibleError("Artifact lead order does not match the required 12-lead order")
    dataset = metadata.get("dataset", {})
    dataset_version = str(dataset.get("version", ""))
    if dataset_version and dataset_version != PTBXL_DATASET_VERSION:
        raise ArtifactIncompatibleError(
            f"Artifact was trained on PTB-XL {dataset_version}; this code targets {PTBXL_DATASET_VERSION}"
        )

    model_config_data = checkpoint.get("model_config") or metadata.get("model_config")
    if not isinstance(model_config_data, dict):
        raise ArtifactIncompatibleError("Artifact has no model configuration")
    try:
        model_config = UnifiedModelConfig.from_dict(model_config_data)
    except (TypeError, ValueError) as exc:
        raise ArtifactIncompatibleError(f"Invalid model configuration: {exc}") from exc
    if model_config.input_channels != len(LEAD_ORDER):
        raise ArtifactIncompatibleError("Model input channel count is not 12")
    if model_config.sequence_length != preprocessing.sequence_length:
        raise ArtifactIncompatibleError(
            "Model sequence length and preprocessing sequence length disagree"
        )

    model = UnifiedECGModel(model_config)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ArtifactIncompatibleError(f"Checkpoint state dict does not match the model: {exc}") from exc
    model.to(device)
    model.eval()

    return LoadedUnifiedArtifacts(
        model=model,
        model_config=model_config,
        preprocessing=preprocessing,
        normalization=normalization,
        metadata=metadata,
        thresholds=thresholds,
        paths=paths,
    )


def default_threshold_payload(
    anomaly_threshold: float,
    class_thresholds: Optional[Sequence[float]] = None,
    *,
    anomaly_info: Optional[Mapping[str, Any]] = None,
    class_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "label_order": list(TARGET_CATEGORIES),
        "anomaly_threshold": float(anomaly_threshold),
        "anomaly_threshold_method": dict(anomaly_info or {}),
        "class_thresholds": [
            float(value) for value in (class_thresholds or [0.5] * len(TARGET_CATEGORIES))
        ],
        "class_threshold_method": dict(class_info or {"method": "default_0.5"}),
        "tuned_on": "PTB-XL validation fold 9 only",
        "frozen": False,
    }


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "MODEL_VERSION",
    "DEFAULT_ARTIFACT_ROOT",
    "ArtifactError",
    "ArtifactNotFoundError",
    "ArtifactIncompatibleError",
    "ArtifactPaths",
    "LoadedUnifiedArtifacts",
    "artifact_paths",
    "resolve_artifact_dir",
    "read_artifact_summary",
    "save_unified_artifacts",
    "write_thresholds",
    "default_threshold_payload",
    "load_unified_artifacts",
    "write_latest_pointer",
]
