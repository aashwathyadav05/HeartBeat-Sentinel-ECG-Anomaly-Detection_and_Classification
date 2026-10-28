"""Metric and threshold utilities for the unified PTB-XL model.

Threshold selection must be run on validation data only. The evaluation
module refuses to report final test results as frozen unless the threshold
artifact was marked frozen beforehand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ptbxl_data import TARGET_CATEGORIES


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    if len(np.unique(y_true)) < 2:
        return None, "AUROC undefined: only one class present in evaluated records"
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, scores)), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"AUROC unavailable: {exc}"


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    if y_true.sum() == 0:
        return None, "Average precision undefined: no positive records"
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, scores)), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Average precision unavailable: {exc}"


def anomaly_metrics(
    references: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Anomaly metrics at a frozen threshold.

    ``references``: 0 = normal reference, 1 = abnormal reference, NaN = unusable.
    ``scores``: reconstruction errors (higher = more anomalous).
    """
    refs = np.asarray(references, dtype=float)
    values = np.asarray(scores, dtype=float)
    if refs.shape != values.shape:
        raise ValueError("references and scores must have the same shape")
    valid = np.isfinite(refs) & np.isfinite(values)
    refs = refs[valid].astype(int)
    values = values[valid]
    notes: List[str] = []
    if len(values) == 0:
        return {
            "threshold": float(threshold),
            "n_evaluated": 0,
            "n_excluded_unusable_labels": int((~valid).sum()),
            "undefined": ["No records with finite scores and usable anomaly reference labels"],
        }
    predicted = (values > float(threshold)).astype(int)
    tp = int(np.sum((refs == 1) & (predicted == 1)))
    tn = int(np.sum((refs == 0) & (predicted == 0)))
    fp = int(np.sum((refs == 0) & (predicted == 1)))
    fn = int(np.sum((refs == 1) & (predicted == 0)))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    accuracy = _safe_divide(tp + tn, len(refs))
    auroc, auroc_note = _auroc(refs, values)
    if auroc_note:
        notes.append(auroc_note)
    for name, value in (("precision", precision), ("recall", recall), ("F1", f1)):
        if value is None:
            notes.append(f"{name} undefined at this threshold (zero denominator)")
    return {
        "threshold": float(threshold),
        "n_evaluated": int(len(refs)),
        "n_excluded_unusable_labels": int((~valid).sum()),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "auroc": auroc,
        "notes": notes,
        "definition": (
            "Abnormal reference = any MI/STTC/CD/HYP label; normal reference = NORM only. "
            "This proxy omits abnormalities outside the five target categories."
        ),
    }


def classification_metrics(
    labels: np.ndarray,
    label_mask: np.ndarray,
    scores: np.ndarray,
    thresholds: Sequence[float],
    *,
    label_order: Sequence[str] = TARGET_CATEGORIES,
) -> Dict[str, Any]:
    """Per-category multilabel metrics using only valid label entries."""
    labels = np.asarray(labels, dtype=float)
    mask = np.asarray(label_mask, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != mask.shape or labels.shape != scores.shape:
        raise ValueError("labels, label_mask and scores must have the same shape")
    if labels.shape[1] != len(label_order):
        raise ValueError("label count does not match label_order")
    if len(thresholds) != len(label_order):
        raise ValueError("one threshold is required per category")
    per_category: Dict[str, Any] = {}
    defined_f1: List[float] = []
    micro_tp = micro_fp = micro_fn = 0
    for index, category in enumerate(label_order):
        valid = mask[:, index] & np.isfinite(labels[:, index]) & np.isfinite(scores[:, index])
        y = labels[valid, index].astype(int)
        s = scores[valid, index]
        threshold = float(thresholds[index])
        predicted = (s >= threshold).astype(int)
        tp = int(np.sum((y == 1) & (predicted == 1)))
        tn = int(np.sum((y == 0) & (predicted == 0)))
        fp = int(np.sum((y == 0) & (predicted == 1)))
        fn = int(np.sum((y == 1) & (predicted == 0)))
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
        auroc, auroc_note = _auroc(y, s)
        average_precision, ap_note = _average_precision(y, s)
        undefined: List[str] = []
        if precision is None:
            undefined.append("precision: no predicted positives")
        if recall is None:
            undefined.append("recall: no positive support")
        if f1 is None:
            undefined.append("F1: no positive support and no predicted positives")
        if auroc_note:
            undefined.append(auroc_note)
        if ap_note:
            undefined.append(ap_note)
        if f1 is not None:
            defined_f1.append(f1)
        per_category[category] = {
            "threshold": threshold,
            "n_evaluated": int(valid.sum()),
            "positive_support": int(y.sum()),
            "predicted_positive_count": int(predicted.sum()),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "auroc": auroc,
            "average_precision": average_precision,
            "undefined": undefined,
        }
    macro_f1 = float(np.mean(defined_f1)) if defined_f1 else None
    micro_f1 = _safe_divide(2 * micro_tp, 2 * micro_tp + micro_fp + micro_fn)
    return {
        "label_order": list(label_order),
        "thresholds": [float(value) for value in thresholds],
        "per_category": per_category,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "categories_with_defined_f1": len(defined_f1),
        "notes": [
            "Macro-F1 averages only categories with a defined F1.",
            "Metrics use only entries whose label mask is valid; unknown labels are never counted as negative.",
            "Sigmoid scores are not calibrated clinical probabilities.",
        ],
    }


def apply_class_thresholds(
    scores: Mapping[str, float] | np.ndarray,
    thresholds: Mapping[str, float] | Sequence[float],
    label_order: Sequence[str] = TARGET_CATEGORIES,
) -> Dict[str, bool]:
    score_values = (
        [float(scores[category]) for category in label_order]
        if isinstance(scores, Mapping)
        else [float(value) for value in scores]
    )
    threshold_values = (
        [float(thresholds[category]) for category in label_order]
        if isinstance(thresholds, Mapping)
        else [float(value) for value in thresholds]
    )
    return {
        category: score_values[index] >= threshold_values[index]
        for index, category in enumerate(label_order)
    }


def select_anomaly_threshold(
    normal_errors: np.ndarray,
    *,
    abnormal_errors: Optional[np.ndarray] = None,
    method: str = "percentile",
    percentile: float = 95.0,
    factor: float = 1.5,
) -> Tuple[float, Dict[str, Any]]:
    """Select the anomaly threshold on validation data only.

    * ``percentile`` uses validation-normal reconstruction errors (default).
    * ``mean_std`` uses mean + factor * std of validation normals.
    * ``f1`` uses validation abnormal records too; it is still validation-only.
    """
    errors = np.asarray(normal_errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        raise ValueError("No finite validation-normal reconstruction errors for threshold selection")
    info: Dict[str, Any] = {
        "method": method,
        "tuned_on": "validation fold 9",
        "normal_count": int(len(errors)),
        "normal_mean": float(errors.mean()),
        "normal_std": float(errors.std()),
    }
    if method == "percentile":
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between 0 and 100")
        threshold = float(np.percentile(errors, percentile))
        info["percentile"] = float(percentile)
    elif method == "mean_std":
        threshold = float(errors.mean() + factor * errors.std())
        info["factor"] = float(factor)
    elif method == "f1":
        if abnormal_errors is None:
            raise ValueError("f1 threshold selection requires validation abnormal errors")
        abnormal = np.asarray(abnormal_errors, dtype=float)
        abnormal = abnormal[np.isfinite(abnormal)]
        if len(abnormal) == 0:
            raise ValueError("No finite validation-abnormal reconstruction errors for f1 selection")
        candidates = np.unique(np.concatenate([errors, abnormal]))
        best_threshold = float(np.median(errors))
        best_f1 = -1.0
        for candidate in candidates:
            predicted_normal = errors > candidate
            predicted_abnormal = abnormal > candidate
            tp = int(predicted_abnormal.sum())
            fp = int(predicted_normal.sum())
            fn = int((~predicted_abnormal).sum())
            denominator = 2 * tp + fp + fn
            f1 = (2 * tp / denominator) if denominator else 0.0
            if f1 > best_f1:
                best_f1 = float(f1)
                best_threshold = float(candidate)
        threshold = best_threshold
        info.update({"abnormal_count": int(len(abnormal)), "validation_f1": best_f1})
    else:
        raise ValueError("method must be 'percentile', 'mean_std' or 'f1'")
    info["threshold"] = float(threshold)
    return float(threshold), info


def _best_f1_threshold(y: np.ndarray, scores: np.ndarray, default: float = 0.5) -> Tuple[float, Optional[float]]:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return default, None
    candidates = np.unique(scores)
    if len(candidates) > 400:
        candidates = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, 400)))
    best_threshold = default
    best_f1 = -1.0
    for candidate in candidates:
        predicted = scores >= candidate
        tp = int(np.sum((y == 1) & predicted))
        fp = int(np.sum((y == 0) & predicted))
        fn = int(np.sum((y == 1) & (~predicted)))
        denominator = 2 * tp + fp + fn
        f1 = (2 * tp / denominator) if denominator else 0.0
        if f1 > best_f1 or (np.isclose(f1, best_f1) and abs(candidate - default) < abs(best_threshold - default)):
            best_f1 = float(f1)
            best_threshold = float(candidate)
    return best_threshold, best_f1


def select_class_thresholds(
    labels: np.ndarray,
    label_mask: np.ndarray,
    scores: np.ndarray,
    *,
    default: float = 0.5,
    label_order: Sequence[str] = TARGET_CATEGORIES,
) -> Tuple[List[float], Dict[str, Any]]:
    """Tune one threshold per category on validation data.

    Categories with no positive or no negative validation examples keep the
    default threshold and are explicitly marked undefined.
    """
    labels = np.asarray(labels, dtype=float)
    mask = np.asarray(label_mask, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    thresholds: List[float] = []
    details: Dict[str, Any] = {}
    for index, category in enumerate(label_order):
        valid = mask[:, index] & np.isfinite(labels[:, index]) & np.isfinite(scores[:, index])
        y = labels[valid, index].astype(int)
        s = scores[valid, index]
        threshold, validation_f1 = _best_f1_threshold(y, s, default=default)
        thresholds.append(float(threshold))
        details[category] = {
            "threshold": float(threshold),
            "validation_f1": validation_f1,
            "positive_support": int(y.sum()),
            "n_evaluated": int(valid.sum()),
            "defined": validation_f1 is not None,
            "note": (
                None
                if validation_f1 is not None
                else "Kept default threshold: validation partition lacks both classes for this category"
            ),
        }
    return thresholds, {
        "method": "per_category_validation_f1",
        "tuned_on": "validation fold 9 only",
        "default": float(default),
        "categories": details,
    }
