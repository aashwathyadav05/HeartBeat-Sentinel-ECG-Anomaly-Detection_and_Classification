"""Single inference service for the unified PTB-XL model.

Both Flask (``web/app.py``) and Streamlit (``app.py``) call this service. It
validates the ECG, applies the artifact's exact preprocessing, runs the shared
encoder once, and returns reconstruction-based anomaly information together
with five independent condition scores.

Anomaly detection and condition classification are intentionally separate
outputs. An anomaly is not evidence of MI/STTC/CD/HYP, and a classifier score
is not a diagnosis.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ptbxl_data import (
    LEAD_ORDER,
    TARGET_CATEGORIES,
    PTBXLValidationError,
    normalize_signal,
    validate_ecg_signal,
)
from .unified_artifacts import (
    ArtifactError,
    ArtifactNotFoundError,
    LoadedUnifiedArtifacts,
    load_unified_artifacts,
    read_artifact_summary,
)

CATEGORY_NAMES: Dict[str, str] = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Changes",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "NORM": (
        "No abnormalities covered by these target categories are identified in the "
        "labelled ECG. This does not rule out every possible heart condition."
    ),
    "MI": (
        "ECG findings associated with damage to heart muscle caused by inadequate "
        "blood supply. Model output alone cannot confirm a heart attack."
    ),
    "STTC": "Changes in the ST segment or T wave. These patterns can have multiple causes.",
    "CD": "An ECG pattern associated with how electrical impulses travel through the heart.",
    "HYP": (
        "ECG patterns that may be associated with thickening or enlargement of heart "
        "muscle; an ECG classifier alone cannot confirm this diagnosis."
    ),
}

DISCLAIMER = (
    "Educational research prototype only. This is not a medical device. Anomaly and "
    "condition outputs are not diagnoses, are not calibrated clinical probabilities, "
    "and must not be used for treatment decisions."
)


class UnifiedInferenceError(RuntimeError):
    """Base class for inference errors safe to show to API clients."""


class ModelNotAvailableError(UnifiedInferenceError):
    """No usable unified artifact is available."""


class InvalidECGInputError(UnifiedInferenceError, PTBXLValidationError):
    """The supplied ECG does not meet the model input contract."""


class UnifiedECGInference:
    """Loads one artifact bundle once and serves unified predictions."""

    def __init__(
        self,
        artifact_path: Optional[str] = None,
        *,
        device: Optional[str] = None,
        load_now: bool = True,
    ) -> None:
        self.artifact_path = artifact_path
        self.device_name = device
        self.artifacts: Optional[LoadedUnifiedArtifacts] = None
        self.load_error: Optional[str] = None
        self._lock = threading.Lock()
        if load_now:
            self.load()

    # -- loading ---------------------------------------------------------
    def load(self) -> LoadedUnifiedArtifacts:
        try:
            import torch

            device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
            artifacts = load_unified_artifacts(self.artifact_path, device=device)
        except ArtifactNotFoundError as exc:
            self.artifacts = None
            self.load_error = str(exc)
            raise ModelNotAvailableError(str(exc)) from exc
        except ArtifactError as exc:
            self.artifacts = None
            self.load_error = str(exc)
            raise ModelNotAvailableError(f"Unified model artifact is unusable: {exc}") from exc
        except Exception as exc:  # pragma: no cover - environment failures
            self.artifacts = None
            self.load_error = str(exc)
            raise ModelNotAvailableError(f"Could not load the unified model: {exc}") from exc
        self.artifacts = artifacts
        self.load_error = None
        return artifacts

    @property
    def is_available(self) -> bool:
        return self.artifacts is not None

    def require_loaded(self) -> LoadedUnifiedArtifacts:
        if self.artifacts is None:
            try:
                return self.load()
            except UnifiedInferenceError:
                raise
            except Exception as exc:
                raise ModelNotAvailableError(str(exc)) from exc
        return self.artifacts

    def health(self) -> Dict[str, Any]:
        if self.artifacts is None:
            return {
                "available": False,
                "error": self.load_error or "Unified model artifacts are not loaded",
                "expected_input": {
                    "lead_order": list(LEAD_ORDER),
                    "shape": [12, 1000],
                    "sampling_rate": 100,
                },
            }
        artifacts = self.artifacts
        return {
            "available": True,
            "model_version": artifacts.metadata.get("model_version"),
            "artifact_dir": artifacts.paths.version_dir,
            "label_order": list(TARGET_CATEGORIES),
            "anomaly_threshold": float(artifacts.thresholds["anomaly_threshold"]),
            "class_thresholds": [float(v) for v in artifacts.thresholds["class_thresholds"]],
            "dataset": artifacts.metadata.get("dataset", {}),
            "expected_input": {
                "lead_order": list(LEAD_ORDER),
                "shape": [len(LEAD_ORDER), artifacts.preprocessing.sequence_length],
                "sampling_rate": artifacts.preprocessing.sampling_rate,
                "normalization": artifacts.normalization.mode,
            },
        }

    # -- prediction ------------------------------------------------------
    def analyze(
        self,
        signal: Any,
        *,
        sampling_rate: float = 100,
        include_signal: bool = True,
        source: str = "api",
    ) -> Dict[str, Any]:
        artifacts = self.require_loaded()
        preprocessing = artifacts.preprocessing
        try:
            validated = validate_ecg_signal(
                signal,
                expected_leads=len(LEAD_ORDER),
                expected_length=preprocessing.sequence_length,
                sampling_rate=sampling_rate,
                expected_sampling_rate=preprocessing.sampling_rate,
            )
        except PTBXLValidationError as exc:
            raise InvalidECGInputError(str(exc)) from exc
        normalized = normalize_signal(validated, artifacts.normalization)

        import torch

        device = next(artifacts.model.parameters()).device
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
        with self._lock:
            with torch.no_grad():
                output = artifacts.model(tensor)
                logits = output.classification_logits[0].detach().cpu().numpy()
                reconstruction = output.reconstruction[0].detach().cpu().numpy()
                scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        return self._format_response(
            original=validated,
            normalized=normalized,
            reconstruction=reconstruction,
            logits=logits,
            scores=scores,
            artifacts=artifacts,
            include_signal=include_signal,
            source=source,
        )

    def _format_response(
        self,
        *,
        original: np.ndarray,
        normalized: np.ndarray,
        reconstruction: np.ndarray,
        logits: np.ndarray,
        scores: np.ndarray,
        artifacts: LoadedUnifiedArtifacts,
        include_signal: bool,
        source: str,
    ) -> Dict[str, Any]:
        squared_error = np.square(normalized - reconstruction)
        pointwise = squared_error.astype(np.float64)
        per_lead = pointwise.mean(axis=1)
        anomaly_score = float(pointwise.mean())
        anomaly_threshold = float(artifacts.thresholds["anomaly_threshold"])
        anomaly_status = "anomalous" if anomaly_score > anomaly_threshold else "normal"
        class_thresholds = [float(v) for v in artifacts.thresholds["class_thresholds"]]
        predicted = [float(score) >= threshold for score, threshold in zip(scores, class_thresholds)]
        predicted_categories = [
            category for category, flag in zip(TARGET_CATEGORIES, predicted) if flag
        ]
        condition_items = []
        for index, category in enumerate(TARGET_CATEGORIES):
            condition_items.append(
                {
                    "category": category,
                    "name": CATEGORY_NAMES[category],
                    "description": CATEGORY_DESCRIPTIONS[category],
                    "score": float(scores[index]),
                    "logit": float(logits[index]),
                    "threshold": class_thresholds[index],
                    "predicted": bool(predicted[index]),
                    "score_type": "sigmoid output (not a calibrated clinical probability)",
                }
            )
        metadata = {
            "model_version": artifacts.metadata.get("model_version"),
            "architecture": artifacts.metadata.get("architecture"),
            "artifact_dir": artifacts.paths.version_dir,
            "dataset": artifacts.metadata.get("dataset", {}),
            "preprocessing": artifacts.metadata.get("preprocessing", {}),
            "thresholds_file": artifacts.paths.thresholds_path,
            "label_order": list(TARGET_CATEGORIES),
            "input_lead_order": list(LEAD_ORDER),
            "input_shape": [len(LEAD_ORDER), artifacts.preprocessing.sequence_length],
            "sampling_rate": artifacts.preprocessing.sampling_rate,
            "source": source,
        }
        response: Dict[str, Any] = {
            "task": "unified_ptbxl",
            "anomaly_score": anomaly_score,
            "anomaly_threshold": anomaly_threshold,
            "anomaly_status": anomaly_status,
            "anomaly": {
                "score": anomaly_score,
                "threshold": anomaly_threshold,
                "status": anomaly_status,
                "per_lead_scores": [float(v) for v in per_lead],
                "lead_order": list(LEAD_ORDER),
                "definition": (
                    "Mean squared reconstruction error over all 12 leads and samples, in "
                    "training-normalized units. It measures deviation from the learned "
                    "normal-ECG reconstruction; it does not identify a condition."
                ),
            },
            "category_scores": {
                category: float(scores[index])
                for index, category in enumerate(TARGET_CATEGORIES)
            },
            "category_thresholds": {
                category: class_thresholds[index]
                for index, category in enumerate(TARGET_CATEGORIES)
            },
            "predicted_categories": predicted_categories,
            "conditions": {
                "label_order": list(TARGET_CATEGORIES),
                "items": condition_items,
                "scores_are_independent": True,
                "score_meaning": (
                    "Each score comes from an independent sigmoid output. Categories can "
                    "coexist. Scores are not calibrated probabilities."
                ),
            },
            "metadata": metadata,
            "disclaimer": DISCLAIMER,
            "notes": [
                "Anomaly detection and condition classification are separate outputs.",
                "An anomalous reconstruction does not imply MI, STTC, CD or HYP.",
                "A predicted category is a research-model output, not a clinical diagnosis.",
            ],
        }
        if include_signal:
            response["reconstruction"] = {
                "original": original.astype(float).tolist(),
                "normalized_input": normalized.astype(float).tolist(),
                "reconstructed": reconstruction.astype(float).tolist(),
                "pointwise_squared_error": pointwise.tolist(),
                "lead_order": list(LEAD_ORDER),
                "units_note": (
                    "original values are the validated input in PTB-XL physical units (mV); "
                    "reconstruction and error are in training-normalized units"
                ),
            }
        return response

    # -- explainability --------------------------------------------------
    def explain(
        self,
        signal: Any,
        category: str,
        *,
        sampling_rate: float = 100,
        window_samples: int = 100,
        stride_samples: int = 50,
    ) -> Dict[str, Any]:
        artifacts = self.require_loaded()
        category = category.strip().upper()
        if category not in TARGET_CATEGORIES:
            raise InvalidECGInputError(f"category must be one of {list(TARGET_CATEGORIES)}")
        try:
            validated = validate_ecg_signal(
                signal,
                expected_length=artifacts.preprocessing.sequence_length,
                sampling_rate=sampling_rate,
                expected_sampling_rate=artifacts.preprocessing.sampling_rate,
            )
        except PTBXLValidationError as exc:
            raise InvalidECGInputError(str(exc)) from exc
        normalized = normalize_signal(validated, artifacts.normalization)
        from .unified_explain import occlusion_sensitivity

        result = occlusion_sensitivity(
            artifacts.model,
            normalized,
            category,
            window_samples=window_samples,
            stride_samples=stride_samples,
            device=next(artifacts.model.parameters()).device,
        )
        result["disclaimer"] = (
            "Occlusion sensitivity highlights temporal regions that influence this model "
            "output. It is not evidence of causal physiology and must not be read as a "
            "clinical explanation."
        )
        return result

    # -- file input ------------------------------------------------------
    def analyze_file(self, path: str, *, filename: Optional[str] = None) -> Dict[str, Any]:
        signal = load_signal_file(path, filename=filename)
        return self.analyze(signal, source="upload")


def load_signal_file(path: str, *, filename: Optional[str] = None) -> np.ndarray:
    """Read a supported ECG file (``.npy``, ``.csv``, ``.txt``, ``.json``)."""
    name = (filename or path).lower()
    if not os.path.isfile(path):
        raise InvalidECGInputError(f"Uploaded file not found: {path}")
    try:
        if name.endswith(".npy"):
            array = np.load(path, allow_pickle=False)
        elif name.endswith(".json"):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                if "signal" not in payload:
                    raise InvalidECGInputError("JSON upload must contain a 'signal' field")
                payload = payload["signal"]
            array = np.asarray(payload)
        elif name.endswith(".csv"):
            array = np.loadtxt(path, delimiter=",", dtype=np.float64)
        elif name.endswith(".txt"):
            array = np.loadtxt(path, dtype=np.float64)
        else:
            raise InvalidECGInputError(
                "Unsupported upload type. Use .npy, .csv, .txt or .json containing one 12x1000 ECG."
            )
    except InvalidECGInputError:
        raise
    except Exception as exc:
        raise InvalidECGInputError(f"Could not parse ECG upload: {exc}") from exc
    try:
        return validate_ecg_signal(array)
    except PTBXLValidationError as exc:
        raise InvalidECGInputError(str(exc)) from exc


_SERVICE: Optional[UnifiedECGInference] = None
_SERVICE_LOCK = threading.Lock()


def get_unified_service(force_reload: bool = False) -> UnifiedECGInference:
    """Process-wide singleton used by both web frontends."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None or force_reload:
            _SERVICE = UnifiedECGInference(load_now=True)
        return _SERVICE


def try_get_unified_service() -> Tuple[Optional[UnifiedECGInference], Optional[str]]:
    """Return the singleton or ``(None, clear_error)`` without raising."""
    try:
        service = get_unified_service()
    except UnifiedInferenceError as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - defensive for web startup
        return None, str(exc)
    if not service.is_available:
        return None, service.load_error or "Unified model is not available"
    return service, None


def summary_or_error() -> Dict[str, Any]:
    """Small health payload that never raises."""
    try:
        return {"available": True, **read_artifact_summary()}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
