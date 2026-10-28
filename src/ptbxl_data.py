"""PTB-XL data access, diagnostic label mapping, preprocessing and splits.

This module is intentionally separate from :mod:`src.data_loader` (ECG5000).
PTB-XL signals are 12-lead, 10-second recordings and must not be routed through
the 140-sample ECG5000 autoencoder.

Design notes
------------
* Target order is fixed: ``[NORM, MI, STTC, CD, HYP]``.
* Labels are multilabel. A record may have several positive categories.
* Only *diagnostic* SCP statements (``diagnostic == 1`` in
  ``scp_statements.csv``) are mapped to the five diagnostic superclasses.
* A record is usable for supervised training only if at least one diagnostic
  statement maps to one of the five target superclasses (and passes
  ``min_likelihood``). For such a record, absence of a target category is
  treated as an annotated negative, because the reporting cardiologist
  supplied a diagnostic statement set. Records with no diagnostic statement,
  only non-diagnostic statements, or only diagnostic statements outside the
  five target classes get an all-False mask and are excluded; they are never
  treated as negatives.
* Patient-aware partitions use the official ``strat_fold`` column:
  folds 1-8 = train, fold 9 = validation, fold 10 = test.
* Normalization statistics are fitted on training-partition *normal* records
  only and stored in model artifacts for exact inference-time reuse.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PTBXL_DATASET_VERSION = "1.0.3"
TARGET_CATEGORIES: Tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
LEAD_ORDER: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
SUPPORTED_SAMPLING_RATES: Tuple[int, ...] = (100, 500)
TRAIN_FOLDS: Tuple[int, ...] = tuple(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10


class PTBXLDataError(RuntimeError):
    """Base class for PTB-XL data problems."""


class PTBXLValidationError(PTBXLDataError):
    """The data or input ECG is malformed or unsupported."""


class PatientLeakageError(PTBXLValidationError):
    """A patient ID occurs in more than one partition."""


class MissingDependencyError(PTBXLDataError):
    """An optional dependency (e.g. wfdb) is unavailable."""


# ---------------------------------------------------------------------------
# Configuration and normalization
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Exact signal contract used by training and inference."""

    sampling_rate: int = 100
    sequence_length: int = 1000
    lead_order: Tuple[str, ...] = LEAD_ORDER
    normalization: str = "train_lead_zscore"
    clip_value: float = 5.0

    def __post_init__(self) -> None:
        if self.sampling_rate not in SUPPORTED_SAMPLING_RATES:
            raise ValueError(f"sampling_rate must be one of {SUPPORTED_SAMPLING_RATES}")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if tuple(self.lead_order) != LEAD_ORDER:
            raise ValueError(f"lead_order must be exactly {LEAD_ORDER}")
        if self.normalization not in {"train_lead_zscore", "per_record_zscore", "none"}:
            raise ValueError("normalization must be 'train_lead_zscore', 'per_record_zscore' or 'none'")
        if self.clip_value <= 0:
            raise ValueError("clip_value must be positive")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["lead_order"] = list(self.lead_order)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreprocessingConfig":
        return cls(
            sampling_rate=int(data.get("sampling_rate", 100)),
            sequence_length=int(data.get("sequence_length", 1000)),
            lead_order=tuple(data.get("lead_order", LEAD_ORDER)),
            normalization=str(data.get("normalization", "train_lead_zscore")),
            clip_value=float(data.get("clip_value", 5.0)),
        )


@dataclass
class NormalizationStats:
    """Per-lead normalization statistics fitted on training normals."""

    mode: str = "train_lead_zscore"
    mean: List[float] = field(default_factory=lambda: [0.0] * len(LEAD_ORDER))
    std: List[float] = field(default_factory=lambda: [1.0] * len(LEAD_ORDER))
    clip_value: float = 5.0

    def __post_init__(self) -> None:
        if len(self.mean) != len(LEAD_ORDER) or len(self.std) != len(LEAD_ORDER):
            raise ValueError(f"Normalization stats need {len(LEAD_ORDER)} values per lead")
        if any((not np.isfinite(v)) for v in self.mean):
            raise ValueError("Normalization mean must be finite")
        if any((not np.isfinite(v) or v <= 0) for v in self.std):
            raise ValueError("Normalization std must be finite and positive")
        if not np.isfinite(self.clip_value) or self.clip_value <= 0:
            raise ValueError("clip_value must be finite and positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizationStats":
        return cls(
            mode=str(data.get("mode", "train_lead_zscore")),
            mean=[float(v) for v in data["mean"]],
            std=[float(v) for v in data["std"]],
            clip_value=float(data.get("clip_value", 5.0)),
        )


def fit_lead_normalization(
    signals: Iterable[np.ndarray],
    clip_value: float = 5.0,
    max_records: Optional[int] = None,
) -> NormalizationStats:
    """Fit per-lead mean/std over an iterable of ``(12, L)`` signals.

    Only training-partition normal records may be passed here. The fit is
    streaming, so the full dataset is never held in memory.
    """
    count = 0
    n_samples = 0
    total = np.zeros(len(LEAD_ORDER), dtype=np.float64)
    total_sq = np.zeros(len(LEAD_ORDER), dtype=np.float64)
    for signal in signals:
        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != len(LEAD_ORDER):
            raise PTBXLValidationError(
                f"Expected a (12, samples) signal while fitting normalization, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise PTBXLValidationError("Cannot fit normalization on non-finite signal values")
        total += arr.sum(axis=1, dtype=np.float64)
        total_sq += np.square(arr).sum(axis=1, dtype=np.float64)
        n_samples += arr.shape[1]
        count += 1
        if max_records is not None and count >= max_records:
            break
    if count == 0 or n_samples == 0:
        raise PTBXLValidationError("No signals supplied for normalization fitting")
    mean = total / n_samples
    variance = np.maximum(total_sq / n_samples - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return NormalizationStats(
        mode="train_lead_zscore",
        mean=mean.tolist(),
        std=std.tolist(),
        clip_value=float(clip_value),
    )


def normalize_signal(signal: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply the saved normalization to one ``(12, L)`` signal."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.shape[0] != len(LEAD_ORDER):
        raise PTBXLValidationError(f"Expected 12 leads, got {arr.shape}")
    if stats.mode == "none":
        return arr.copy()
    if stats.mode == "per_record_zscore":
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        out = (arr - mean) / std
    elif stats.mode == "train_lead_zscore":
        mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1, 1)
        std = np.asarray(stats.std, dtype=np.float32).reshape(-1, 1)
        out = (arr - mean) / std
    else:  # pragma: no cover - guarded by dataclass
        raise PTBXLValidationError(f"Unknown normalization mode: {stats.mode}")
    return np.clip(out, -stats.clip_value, stats.clip_value).astype(np.float32)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_ecg_signal(
    signal: Any,
    *,
    expected_leads: int = len(LEAD_ORDER),
    expected_length: int = 1000,
    sampling_rate: Optional[float] = None,
    expected_sampling_rate: Optional[int] = 100,
) -> np.ndarray:
    """Validate and return a float32 ``(12, 1000)`` ECG array.

    Deliberately does **not** transpose, pad, truncate or reshape a signal with
    a different shape. A 140-sample ECG5000 vector is rejected with a message
    pointing at the legacy workflow.
    """
    try:
        arr = np.asarray(signal)
    except Exception as exc:  # pragma: no cover - exotic inputs
        raise PTBXLValidationError(f"Could not convert ECG input to an array: {exc}") from exc
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape == (140,) or arr.shape == (1, 140):
        raise PTBXLValidationError(
            "Received a 140-sample ECG5000 signal. That input belongs to the legacy "
            "ECG5000 workflow and cannot be routed into the 12-lead PTB-XL model."
        )
    if arr.ndim != 2:
        raise PTBXLValidationError(
            f"ECG input must be a 2-D (leads, samples) array, got shape {arr.shape}"
        )
    if arr.shape[0] != expected_leads:
        raise PTBXLValidationError(
            f"ECG input must contain exactly {expected_leads} leads in order "
            f"{list(LEAD_ORDER)}; got {arr.shape[0]} rows with shape {arr.shape}."
        )
    if arr.shape[1] != expected_length:
        raise PTBXLValidationError(
            f"ECG input must contain exactly {expected_length} samples per lead; got {arr.shape[1]}."
        )
    if arr.dtype == bool or arr.dtype.kind in {"U", "S", "O"}:
        raise PTBXLValidationError("ECG input values must be numeric (not strings, objects or booleans)")
    try:
        arr = arr.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise PTBXLValidationError(f"ECG input values must be numeric: {exc}") from exc
    if not np.all(np.isfinite(arr)):
        raise PTBXLValidationError("ECG input contains NaN or infinite values")
    if sampling_rate is not None and expected_sampling_rate is not None:
        if not np.isclose(float(sampling_rate), float(expected_sampling_rate)):
            raise PTBXLValidationError(
                f"Expected a {expected_sampling_rate} Hz ECG, got {sampling_rate} Hz. "
                "No implicit resampling is performed."
            )
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Metadata and label mapping
# ---------------------------------------------------------------------------


def _parse_scp_codes(value: Any) -> Tuple[Dict[str, Any], bool]:
    if isinstance(value, dict):
        return value, True
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}, False
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}, False
        if isinstance(parsed, dict):
            return parsed, True
    return {}, False


def load_ptbxl_metadata(
    data_dir: str,
    filename: str = "ptbxl_database.csv",
) -> pd.DataFrame:
    """Load and validate the official PTB-XL metadata CSV."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise PTBXLDataError(
            f"PTB-XL metadata not found: {path}. Download PTB-XL 1.0.3 from "
            "https://physionet.org/content/ptb-xl/1.0.3/ and see README for placement."
        )
    df = pd.read_csv(path)
    required = {"ecg_id", "patient_id", "strat_fold", "scp_codes"}
    missing = required - set(df.columns)
    if missing:
        raise PTBXLValidationError(f"{filename} is missing required columns: {sorted(missing)}")
    if df["ecg_id"].isna().any() or not df["ecg_id"].is_unique:
        raise PTBXLValidationError("ecg_id values must be present and unique")
    if df["patient_id"].isna().any():
        raise PTBXLValidationError("patient_id values must be present")
    bad_folds = ~df["strat_fold"].isin(list(range(1, 11)))
    if bad_folds.any():
        LOGGER.warning("Dropping %d record(s) with unsupported strat_fold values", int(bad_folds.sum()))
        df = df.loc[~bad_folds].copy()
    df["patient_id"] = df["patient_id"].astype(int)
    df["ecg_id"] = df["ecg_id"].astype(int)
    df["strat_fold"] = df["strat_fold"].astype(int)

    parsed: List[Dict[str, Any]] = []
    parse_ok: List[bool] = []
    for value in df["scp_codes"]:
        codes, ok = _parse_scp_codes(value)
        parsed.append(codes)
        parse_ok.append(ok)
    df["scp_codes_parsed"] = parsed
    df["scp_codes_parsed_ok"] = parse_ok
    invalid = int((~np.asarray(parse_ok)).sum())
    if invalid:
        LOGGER.warning("Could not parse scp_codes for %d record(s); they will be excluded", invalid)

    if "filename_lr" in df.columns:
        missing_lr = int(df["filename_lr"].isna().sum())
        if missing_lr:
            LOGGER.warning("%d record(s) have no filename_lr (100 Hz) entry", missing_lr)
    df.attrs["n_invalid_scp_codes"] = invalid
    return df.reset_index(drop=True)


def load_scp_statements(
    data_dir: str,
    filename: str = "scp_statements.csv",
) -> pd.DataFrame:
    """Load the official SCP statement dictionary with diagnostic classes."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise PTBXLDataError(
            f"SCP statements file not found: {path}. It is part of the PTB-XL 1.0.3 download."
        )
    statements = pd.read_csv(path, index_col=0)
    if not {"diagnostic", "diagnostic_class"}.issubset(statements.columns):
        raise PTBXLValidationError(
            "scp_statements.csv must contain 'diagnostic' and 'diagnostic_class' columns"
        )
    if statements.index.has_duplicates:
        raise PTBXLValidationError("scp_statements.csv contains duplicate SCP codes")
    statements.index = statements.index.astype(str).str.strip()
    statements["diagnostic_class"] = (
        statements["diagnostic_class"].fillna("").astype(str).str.strip().str.upper()
    )
    return statements


def map_scp_codes_to_targets(
    scp_codes: Mapping[str, Any],
    statements: pd.DataFrame,
    *,
    min_likelihood: float = 0.0,
    target_categories: Sequence[str] = TARGET_CATEGORIES,
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str, float]], bool]:
    """Map one record's SCP codes to the five target categories.

    Returns ``(labels, matched_codes, diagnostic_statements, usable)``.
    ``usable`` is True only when at least one diagnostic statement maps to a
    target category and meets ``min_likelihood``. A diagnostic statement in an
    unmapped class is returned for auditing but does not make the record
    usable, because treating its five target labels as negatives would turn
    an unknown label into a negative label.
    """
    targets = list(target_categories)
    labels = {category: 0.0 for category in targets}
    matched: List[str] = []
    diagnostics: List[Tuple[str, str, float]] = []
    for raw_code, raw_likelihood in dict(scp_codes).items():
        code = str(raw_code).strip()
        if code not in statements.index:
            continue
        row = statements.loc[code]
        try:
            is_diagnostic = float(row.get("diagnostic", 0) or 0) == 1.0
        except (TypeError, ValueError):
            is_diagnostic = False
        if not is_diagnostic:
            continue
        category = str(row.get("diagnostic_class", "") or "").strip().upper()
        try:
            likelihood = float(raw_likelihood)
        except (TypeError, ValueError):
            likelihood = 0.0
        diagnostics.append((code, category, likelihood))
        if category in labels and likelihood >= min_likelihood:
            labels[category] = 1.0
            matched.append(code)
    usable = bool(matched)
    return (
        np.asarray([labels[c] for c in targets], dtype=np.float32),
        matched,
        diagnostics,
        usable,
    )


def anomaly_reference_from_labels(
    labels: np.ndarray,
    label_mask: np.ndarray,
) -> np.ndarray:
    """Derive a binary anomaly reference, or NaN when labels are unusable.

    Normal reference: NORM positive and none of MI/STTC/CD/HYP positive.
    Abnormal reference: any of MI/STTC/CD/HYP positive.
    This is a proxy for "outside the NORM superclass", not a complete
    clinical definition of every possible ECG abnormality.
    """
    labels = np.asarray(labels)
    label_mask = np.asarray(label_mask, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != len(TARGET_CATEGORIES):
        raise PTBXLValidationError("labels must have shape (n, 5)")
    out = np.full(labels.shape[0], np.nan, dtype=np.float32)
    row_valid = label_mask.any(axis=1)
    norm = labels[:, 0] == 1
    other = labels[:, 1:].sum(axis=1) > 0
    out[row_valid & norm & ~other] = 0.0
    out[row_valid & other] = 1.0
    return out


def assign_split_from_fold(fold: int) -> Optional[str]:
    fold = int(fold)
    if 1 <= fold <= 8:
        return "train"
    if fold == VAL_FOLD:
        return "val"
    if fold == TEST_FOLD:
        return "test"
    return None


def validate_patient_splits(
    metadata: pd.DataFrame,
    splits: Sequence[str],
    *,
    raise_on_leakage: bool = True,
) -> Dict[str, Any]:
    """Verify that no patient appears in more than one partition."""
    frame = metadata.copy()
    frame["_split"] = list(splits)
    patients_by_split: Dict[str, set] = {}
    summary: Dict[str, Any] = {"counts": {}, "patient_counts": {}, "leakage": {}}
    for split in ("train", "val", "test"):
        subset = frame[frame["_split"] == split]
        summary["counts"][split] = int(len(subset))
        summary["patient_counts"][split] = int(subset["patient_id"].nunique())
        patients_by_split[split] = set(subset["patient_id"].astype(int))
    leakages: Dict[str, List[int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(patients_by_split[left] & patients_by_split[right])
        if overlap:
            leakages[f"{left}-{right}"] = [int(v) for v in overlap[:20]]
    summary["leakage"] = leakages
    if leakages and raise_on_leakage:
        raise PatientLeakageError(
            "Patient leakage detected across official PTB-XL folds: "
            + json.dumps(leakages)
            + ". The official folds should be patient-aware; check the metadata."
        )
    return summary


@dataclass
class PTBXLBundle:
    """Metadata, multilabel targets and mapping statistics."""

    metadata: pd.DataFrame
    labels: np.ndarray
    label_mask: np.ndarray
    statements: pd.DataFrame
    stats: Dict[str, Any]
    dataset_version: str = PTBXL_DATASET_VERSION

    @property
    def splits(self) -> np.ndarray:
        return self.metadata["split"].to_numpy()

    def split_frame(self, split: str) -> pd.DataFrame:
        return self.metadata.loc[self.metadata["split"] == split].copy()

    def split_indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.metadata["split"].to_numpy() == split)

    def usable_frame(self, split: Optional[str] = None) -> pd.DataFrame:
        frame = self.metadata if split is None else self.split_frame(split)
        return frame.loc[frame["has_target_labels"]].copy()


def load_ptbxl_bundle(
    data_dir: str,
    *,
    min_likelihood: float = 0.0,
    metadata_file: str = "ptbxl_database.csv",
    statements_file: str = "scp_statements.csv",
) -> PTBXLBundle:
    """Load metadata, map labels, assign official folds and verify splits."""
    metadata = load_ptbxl_metadata(data_dir, metadata_file)
    statements = load_scp_statements(data_dir, statements_file)

    n = len(metadata)
    labels = np.zeros((n, len(TARGET_CATEGORIES)), dtype=np.float32)
    label_mask = np.zeros_like(labels, dtype=bool)
    matched_codes: List[List[str]] = []
    for position, codes in enumerate(metadata["scp_codes_parsed"]):
        mapped, matched, _diagnostics, usable = map_scp_codes_to_targets(
            codes, statements, min_likelihood=min_likelihood
        )
        labels[position] = mapped
        if usable:
            label_mask[position, :] = True
        matched_codes.append(matched)

    metadata["mapped_scp_codes"] = ["|".join(codes) for codes in matched_codes]
    metadata["has_target_labels"] = label_mask.any(axis=1)
    metadata["anomaly_reference"] = anomaly_reference_from_labels(labels, label_mask)
    metadata["split"] = [assign_split_from_fold(fold) for fold in metadata["strat_fold"]]
    metadata.loc[metadata["split"].isna(), "has_target_labels"] = False

    splits = metadata["split"].fillna("excluded").tolist()
    split_summary = validate_patient_splits(metadata, splits, raise_on_leakage=True)

    usable_mask = label_mask.any(axis=1)
    positives = {
        category: int(labels[usable_mask, idx].sum())
        for idx, category in enumerate(TARGET_CATEGORIES)
    }
    stats: Dict[str, Any] = {
        "dataset_version": PTBXL_DATASET_VERSION,
        "n_records": int(n),
        "n_usable_for_supervision": int(usable_mask.sum()),
        "n_excluded_no_diagnostic_label": int((~usable_mask).sum()),
        "n_mapped_diagnostic_codes": int(sum(len(c) for c in matched_codes)),
        "n_unparsed_scp_codes": int(metadata.attrs.get("n_invalid_scp_codes", 0)),
        "target_order": list(TARGET_CATEGORIES),
        "positive_counts": positives,
        "label_mask_semantics": (
            "A record with at least one mapped diagnostic statement is usable; "
            "absence of a target category is an annotated negative. Records with "
            "no diagnostic statement are excluded and never treated as negative."
        ),
        "split_summary": split_summary,
        "min_likelihood": float(min_likelihood),
    }
    LOGGER.info("PTB-XL records: %d total, %d usable, %d excluded for unusable labels",
                n, stats["n_usable_for_supervision"], stats["n_excluded_no_diagnostic_label"])
    LOGGER.info("Per-category positives among usable records: %s", positives)

    return PTBXLBundle(
        metadata=metadata,
        labels=labels,
        label_mask=label_mask,
        statements=statements,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Signal reading and caching
# ---------------------------------------------------------------------------


def _import_wfdb():
    try:
        import wfdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError(
            "The 'wfdb' package is required to read PTB-XL waveforms. "
            "Install it with: pip install wfdb"
        ) from exc
    return wfdb


def read_ptbxl_signal(
    data_dir: str,
    record_path: str,
    *,
    sampling_rate: int = 100,
    expected_length: int = 1000,
) -> np.ndarray:
    """Read a PTB-XL WFDB record and return float32 ``(12, 1000)``.

    Lead order is enforced; no implicit resampling or transposition of a
    malformed array is performed.
    """
    if sampling_rate not in SUPPORTED_SAMPLING_RATES:
        raise PTBXLValidationError(f"sampling_rate must be one of {SUPPORTED_SAMPLING_RATES}")
    if not record_path:
        raise PTBXLValidationError("Record has no WFDB file path in the metadata")
    record_path = str(record_path).replace("\\", os.sep).replace("/", os.sep)
    full_path = os.path.join(data_dir, record_path)
    # WFDB rdrecord accepts a path without extension.
    root, ext = os.path.splitext(full_path)
    candidates = [full_path] if ext else [full_path]
    existing = [p for p in candidates if os.path.exists(p + ".hea") or os.path.exists(p + ".dat")]
    if not existing:
        raise PTBXLDataError(
            f"WFDB record not found at {full_path}. Expected .hea/.dat files under the "
            "PTB-XL data directory."
        )
    wfdb = _import_wfdb()
    record = wfdb.rdrecord(full_path)
    if record.p_signal is None:
        raise PTBXLValidationError(f"WFDB record {full_path} has no physical signal")
    actual_fs = float(record.fs)
    if not np.isclose(actual_fs, float(sampling_rate)):
        raise PTBXLValidationError(
            f"Record {record_path} has sampling rate {actual_fs} Hz, expected {sampling_rate} Hz. "
            "No implicit resampling is performed; choose the matching PTB-XL filename column."
        )
    names = [str(name).strip() for name in (record.sig_name or [])]
    canonical_by_casefold = {lead.casefold(): lead for lead in LEAD_ORDER}
    positions: Dict[str, int] = {}
    duplicates: set[str] = set()
    unexpected: List[str] = []
    for index, name in enumerate(names):
        canonical = canonical_by_casefold.get(name.casefold())
        if canonical is None:
            unexpected.append(name)
            continue
        if canonical in positions:
            duplicates.add(canonical)
        else:
            positions[canonical] = index

    missing = [lead for lead in LEAD_ORDER if lead not in positions]
    if missing or duplicates or unexpected or len(names) != len(LEAD_ORDER):
        raise PTBXLValidationError(
            f"WFDB record {record_path} has invalid lead names: "
            f"missing={missing}, duplicate={sorted(duplicates)}, unexpected={unexpected}, "
            f"count={len(names)}. Expected exactly one of each lead, case-insensitively, "
            f"in {list(LEAD_ORDER)}. Check the record header and lead mapping."
        )
    column_order = [positions[lead] for lead in LEAD_ORDER]
    # WFDB exposes p_signal as (samples, signals); the rest of this pipeline
    # uses (leads, samples).
    signal = np.asarray(record.p_signal, dtype=np.float32)[:, column_order].T
    if signal.shape != (len(LEAD_ORDER), expected_length):
        raise PTBXLValidationError(
            f"Record {record_path} produced shape {signal.shape}; expected "
            f"({len(LEAD_ORDER)}, {expected_length})"
        )
    if not np.all(np.isfinite(signal)):
        raise PTBXLValidationError(f"Record {record_path} contains NaN or infinite values")
    return np.ascontiguousarray(signal, dtype=np.float32)


def record_column_for_rate(sampling_rate: int) -> str:
    if sampling_rate == 100:
        return "filename_lr"
    if sampling_rate == 500:
        return "filename_hr"
    raise PTBXLValidationError(f"Unsupported sampling rate {sampling_rate}")


def build_signal_cache(
    bundle: PTBXLBundle,
    data_dir: str,
    cache_dir: str,
    *,
    sampling_rate: int = 100,
    expected_length: int = 1000,
    max_records: Optional[int] = None,
    record_reader: Callable[..., np.ndarray] = read_ptbxl_signal,
) -> Dict[str, Any]:
    """Read WFDB records once into a disk-backed float32 array.

    Invalid records are skipped and logged; their ecg_ids are written to the
    manifest so training can exclude them explicitly.
    """
    column = record_column_for_rate(sampling_rate)
    if column not in bundle.metadata.columns:
        raise PTBXLValidationError(f"Metadata has no '{column}' column for {sampling_rate} Hz signals")
    os.makedirs(cache_dir, exist_ok=True)
    frames = bundle.metadata
    if max_records is not None:
        frames = frames.iloc[: int(max_records)]
    if len(frames) == 0:
        raise PTBXLDataError("No metadata records were selected for signal-cache construction")
    signals_path = os.path.join(cache_dir, "signals.npy")
    staging_path = os.path.join(cache_dir, "signals_staging.npy")
    shape = (len(frames), len(LEAD_ORDER), expected_length)
    array = np.lib.format.open_memmap(staging_path, mode="w+", dtype=np.float32, shape=shape)
    keep_positions: List[int] = []
    kept_ids: List[int] = []
    skipped: List[Dict[str, Any]] = []
    for position in range(len(frames)):
        row = frames.iloc[position]
        try:
            signal = record_reader(
                data_dir,
                row[column],
                sampling_rate=sampling_rate,
                expected_length=expected_length,
            )
            array[len(keep_positions)] = signal
            keep_positions.append(position)
            kept_ids.append(int(row["ecg_id"]))
        except Exception as exc:  # noqa: BLE001 - dataset audit must continue
            skipped.append({"ecg_id": int(row["ecg_id"]), "reason": str(exc)})
            LOGGER.warning("Skipping ecg_id %s: %s", row["ecg_id"], exc)
    array.flush()
    del array
    if keep_positions:
        cached = np.load(staging_path, mmap_mode="r")[: len(keep_positions)]
        np.save(signals_path, np.asarray(cached))
        del cached
    else:
        np.save(signals_path, np.empty((0, len(LEAD_ORDER), expected_length), dtype=np.float32))
    try:
        os.remove(staging_path)
    except OSError:
        pass
    manifest = {
        "dataset_version": PTBXL_DATASET_VERSION,
        "sampling_rate": sampling_rate,
        "sequence_length": expected_length,
        "lead_order": list(LEAD_ORDER),
        "ecg_ids": kept_ids,
        "n_attempted": int(len(frames)),
        "n_cached": len(kept_ids),
        "n_failed": len(skipped),
        "n_skipped": len(skipped),
        "skipped": skipped,
    }
    manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    if not kept_ids:
        raise PTBXLDataError(
            f"No signals were cached (0/{len(frames)} records succeeded); "
            f"inspect skipped-record details in {manifest_path}"
        )
    return manifest


def read_signal_cache(cache_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    signals_path = os.path.join(cache_dir, "signals.npy")
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(signals_path) or not os.path.exists(manifest_path):
        raise PTBXLDataError(f"No signal cache found in {cache_dir}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    array = np.load(signals_path, mmap_mode="r")
    if array.ndim != 3 or array.shape[1] != len(LEAD_ORDER):
        raise PTBXLValidationError(f"Cached signals have unexpected shape {array.shape}")
    cached_ids = manifest.get("ecg_ids", [])
    if len(cached_ids) != array.shape[0]:
        raise PTBXLValidationError(
            f"Cache manifest lists {len(cached_ids)} record IDs but signals.npy contains "
            f"{array.shape[0]} records"
        )
    if array.shape[0] == 0:
        raise PTBXLDataError(f"Signal cache at {cache_dir} contains zero records")
    declared_count = manifest.get("n_cached")
    if declared_count is not None and int(declared_count) != array.shape[0]:
        raise PTBXLValidationError(
            f"Cache manifest n_cached={declared_count} disagrees with signals.npy "
            f"record count {array.shape[0]}"
        )
    return array, manifest


class PTBXLSignalDataset:
    """PyTorch dataset over validated PTB-XL records.

    Returns a dict with normalized signal, multilabel targets, validity mask,
    fold and identifiers. Works from WFDB files or an optional cache.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        labels: np.ndarray,
        label_mask: np.ndarray,
        data_dir: str,
        preprocessing: PreprocessingConfig,
        normalization: Optional[NormalizationStats] = None,
        *,
        cache_dir: Optional[str] = None,
        record_reader: Callable[..., np.ndarray] = read_ptbxl_signal,
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, outer: "PTBXLSignalDataset") -> None:
                self.outer = outer

            def __len__(self) -> int:
                return len(self.outer.metadata)

            def __getitem__(self, index: int) -> Dict[str, Any]:
                return self.outer.get(index)

        self._torch = torch
        self._dataset_class = _Dataset
        self.metadata = metadata.reset_index(drop=True)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.label_mask = np.asarray(label_mask, dtype=bool)
        if len(self.metadata) != len(self.labels) or len(self.metadata) != len(self.label_mask):
            raise PTBXLValidationError("Metadata, labels and label mask must have equal length")
        self.data_dir = data_dir
        self.preprocessing = preprocessing
        self.normalization = normalization or NormalizationStats(
            mode="none", clip_value=preprocessing.clip_value
        )
        self.cache_dir = cache_dir
        self.record_reader = record_reader
        self._signals = None
        self._cache_lookup: Dict[int, int] = {}
        if cache_dir:
            signals, manifest = read_signal_cache(cache_dir)
            if int(manifest.get("sampling_rate", -1)) != preprocessing.sampling_rate:
                raise PTBXLValidationError("Signal cache sampling rate does not match preprocessing config")
            if int(manifest.get("sequence_length", -1)) != preprocessing.sequence_length:
                raise PTBXLValidationError("Signal cache sequence length does not match preprocessing config")
            if list(manifest.get("lead_order", [])) != list(LEAD_ORDER):
                raise PTBXLValidationError("Signal cache lead order does not match the required order")
            self._signals = signals
            self._cache_lookup = {
                int(ecg_id): pos for pos, ecg_id in enumerate(manifest.get("ecg_ids", []))
            }
            missing_ids = [
                int(ecg_id)
                for ecg_id in self.metadata["ecg_id"].astype(int)
                if int(ecg_id) not in self._cache_lookup
            ]
            if missing_ids:
                examples = missing_ids[:10]
                raise PTBXLDataError(
                    f"Signal cache is missing {len(missing_ids)} requested record(s), "
                    f"including ecg_id values {examples}. Review manifest.json and exclude "
                    "or repair skipped records before training."
                )
        self._column = record_column_for_rate(preprocessing.sampling_rate)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index)

    def get(self, index: int) -> Dict[str, Any]:
        row = self.metadata.iloc[index]
        ecg_id = int(row["ecg_id"])
        if self._signals is not None:
            if ecg_id not in self._cache_lookup:
                raise PTBXLDataError(
                    f"ecg_id {ecg_id} is not present in the signal cache (it may have been skipped as invalid)"
                )
            raw = np.asarray(self._signals[self._cache_lookup[ecg_id]], dtype=np.float32)
        else:
            if self._column not in row or pd.isna(row[self._column]):
                raise PTBXLValidationError(f"ecg_id {ecg_id} has no waveform path for the configured sampling rate")
            raw = self.record_reader(
                self.data_dir,
                row[self._column],
                sampling_rate=self.preprocessing.sampling_rate,
                expected_length=self.preprocessing.sequence_length,
            )
        validated = validate_ecg_signal(
            raw,
            expected_leads=len(LEAD_ORDER),
            expected_length=self.preprocessing.sequence_length,
        )
        normalized = normalize_signal(validated, self.normalization)
        signal_tensor = self._torch.from_numpy(normalized)
        return {
            "signal": signal_tensor,
            "labels": self._torch.from_numpy(self.labels[index].copy()),
            "label_mask": self._torch.from_numpy(self.label_mask[index].copy()),
            "ecg_id": ecg_id,
            "patient_id": int(row["patient_id"]),
            "fold": int(row["strat_fold"]),
        }

    def dataset(self):
        """Return the underlying ``torch.utils.data.Dataset`` instance."""
        return self._dataset_class(self)


def make_dataloader(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    label_mask: np.ndarray,
    data_dir: str,
    preprocessing: PreprocessingConfig,
    normalization: NormalizationStats,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    record_reader: Callable[..., np.ndarray] = read_ptbxl_signal,
):
    from torch.utils.data import DataLoader

    dataset = PTBXLSignalDataset(
        metadata,
        labels,
        label_mask,
        data_dir,
        preprocessing,
        normalization,
        cache_dir=cache_dir,
        record_reader=record_reader,
    )
    return DataLoader(dataset.dataset(), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def select_frame(
    bundle: PTBXLBundle,
    split: str,
    *,
    require_labels: bool = False,
    normal_only: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Select records for a partition and return aligned labels/masks."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'")
    positions = bundle.metadata.index[bundle.metadata["split"] == split]
    frame = bundle.metadata.loc[positions]
    labels = bundle.labels[positions]
    mask = bundle.label_mask[positions]
    keep = np.ones(len(frame), dtype=bool)
    if require_labels:
        keep &= frame["has_target_labels"].to_numpy(dtype=bool)
    if normal_only:
        keep &= (frame["anomaly_reference"].to_numpy(dtype=float) == 0.0)
    frame = frame.loc[keep].reset_index(drop=True)
    return frame, labels[keep], mask[keep]


def load_signals_for_frame(
    frame: pd.DataFrame,
    data_dir: str,
    preprocessing: PreprocessingConfig,
    *,
    cache_dir: Optional[str] = None,
    record_reader: Callable[..., np.ndarray] = read_ptbxl_signal,
    max_records: Optional[int] = None,
) -> Iterable[np.ndarray]:
    """Yield raw (un-normalized) signals for a frame."""
    column = record_column_for_rate(preprocessing.sampling_rate)
    cache = None
    lookup: Dict[int, int] = {}
    if cache_dir:
        cache, manifest = read_signal_cache(cache_dir)
        lookup = {int(ecg_id): pos for pos, ecg_id in enumerate(manifest.get("ecg_ids", []))}
    for count, (_idx, row) in enumerate(frame.iterrows()):
        if max_records is not None and count >= max_records:
            break
        ecg_id = int(row["ecg_id"])
        if cache is not None:
            if ecg_id not in lookup:
                raise PTBXLDataError(
                    f"ecg_id {ecg_id} is not present in the signal cache; "
                    "repair or rebuild the cache instead of fitting preprocessing on a subset"
                )
            signal = np.asarray(cache[lookup[ecg_id]], dtype=np.float32)
        else:
            signal = record_reader(
                data_dir,
                row[column],
                sampling_rate=preprocessing.sampling_rate,
                expected_length=preprocessing.sequence_length,
            )
        yield validate_ecg_signal(signal, expected_length=preprocessing.sequence_length)


def artifact_dataset_info(bundle: PTBXLBundle, preprocessing: PreprocessingConfig) -> Dict[str, Any]:
    """Dataset description persisted alongside model artifacts."""
    return {
        "name": "PTB-XL",
        "version": PTBXL_DATASET_VERSION,
        "sampling_rate": preprocessing.sampling_rate,
        "sequence_length": preprocessing.sequence_length,
        "lead_order": list(LEAD_ORDER),
        "fold_strategy": {"train": list(TRAIN_FOLDS), "val": VAL_FOLD, "test": TEST_FOLD},
        "label_order": list(TARGET_CATEGORIES),
        "stats": bundle.stats,
    }


def sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
