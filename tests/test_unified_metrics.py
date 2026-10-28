"""Anomaly/classification metrics and validation-only threshold tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.ptbxl_data import TARGET_CATEGORIES
from src.unified_metrics import (
    anomaly_metrics,
    apply_class_thresholds,
    classification_metrics,
    select_anomaly_threshold,
    select_class_thresholds,
)


def test_anomaly_metrics_mask_unusable_labels_and_report_confusion():
    references = np.asarray([0.0, 0.0, 1.0, 1.0, np.nan])
    scores = np.asarray([0.1, 0.6, 0.4, 0.9, 0.2])
    metrics = anomaly_metrics(references, scores, threshold=0.5)

    assert metrics["n_evaluated"] == 4
    assert metrics["n_excluded_unusable_labels"] == 1
    assert metrics["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert "Abnormal reference" in metrics["definition"]


def test_anomaly_metrics_strict_threshold_and_empty_partition():
    metrics = anomaly_metrics(np.asarray([0.0]), np.asarray([0.5]), threshold=0.5)
    # Strictly greater than the threshold: exactly at threshold is normal.
    assert metrics["confusion"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 0}

    empty = anomaly_metrics(
        np.asarray([np.nan, np.nan]), np.asarray([0.1, 0.2]), threshold=0.5
    )
    assert empty["n_evaluated"] == 0
    assert empty["undefined"]


def test_classification_metrics_never_count_masked_entries_as_negatives():
    labels = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    # The second row is entirely unknown/unusable. Its high MI score must not
    # be counted as a false positive.
    mask = np.asarray(
        [
            [True, True, True, True, True],
            [False, False, False, False, False],
        ]
    )
    scores = np.asarray(
        [
            [0.9, 0.1, 0.2, 0.2, 0.2],
            [0.9, 0.9, 0.9, 0.9, 0.9],
        ]
    )
    metrics = classification_metrics(labels, mask, scores, [0.5] * 5)

    norm = metrics["per_category"]["NORM"]
    mi = metrics["per_category"]["MI"]
    assert norm["n_evaluated"] == 1
    assert norm["confusion"] == {"tn": 0, "fp": 0, "fn": 0, "tp": 1}
    assert mi["n_evaluated"] == 1
    assert mi["predicted_positive_count"] == 0
    assert mi["confusion"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 0}
    assert metrics["label_order"] == list(TARGET_CATEGORIES)
    assert metrics["notes"]


def test_classification_metrics_marks_undefined_categories():
    labels = np.zeros((3, 5), dtype=float)
    mask = np.ones((3, 5), dtype=bool)
    scores = np.full((3, 5), 0.1)
    metrics = classification_metrics(labels, mask, scores, [0.5] * 5)
    assert metrics["macro_f1"] is None
    assert metrics["micro_f1"] is None
    assert metrics["categories_with_defined_f1"] == 0
    for category in TARGET_CATEGORIES:
        assert metrics["per_category"][category]["f1"] is None
        assert metrics["per_category"][category]["undefined"]


def test_metric_shape_validation():
    with pytest.raises(ValueError):
        anomaly_metrics(np.asarray([0.0]), np.asarray([0.0, 1.0]), threshold=0.5)
    with pytest.raises(ValueError):
        classification_metrics(
            np.zeros((2, 5)), np.ones((2, 5)), np.zeros((2, 5)), [0.5] * 4
        )


def test_apply_class_thresholds_mapping_and_sequence():
    labels = list(TARGET_CATEGORIES)
    scores = {category: 0.5 for category in labels}
    thresholds = {category: 0.5 for category in labels}
    result = apply_class_thresholds(scores, thresholds)
    assert all(result.values())

    thresholds["MI"] = 0.6
    result = apply_class_thresholds(scores, thresholds)
    assert result["MI"] is False
    assert result["NORM"] is True

    sequence_result = apply_class_thresholds([0.1, 0.2, 0.3, 0.4, 0.5], [0.1] * 5)
    assert sequence_result == {category: True for category in labels}


def test_select_anomaly_threshold_methods_use_supplied_normals_only():
    normals = np.asarray([1.0, 2.0, 3.0, 4.0])
    threshold, info = select_anomaly_threshold(normals, percentile=50)
    assert threshold == pytest.approx(2.5)
    assert info["tuned_on"] == "validation fold 9"
    assert info["normal_count"] == 4

    threshold, info = select_anomaly_threshold(
        normals, method="mean_std", factor=1.0
    )
    assert threshold == pytest.approx(normals.mean() + normals.std())
    assert info["factor"] == 1.0

    threshold, info = select_anomaly_threshold(
        np.asarray([1.0, 2.0, 3.0]),
        abnormal_errors=np.asarray([4.0, 5.0]),
        method="f1",
    )
    assert 2.0 <= threshold <= 5.0
    assert info["validation_f1"] == pytest.approx(1.0)

    with pytest.raises(ValueError):
        select_anomaly_threshold(np.asarray([]))
    with pytest.raises(ValueError):
        select_anomaly_threshold(normals, method="unknown")
    with pytest.raises(ValueError):
        select_anomaly_threshold(np.asarray([1.0]), method="f1")


def test_select_class_thresholds_tunes_only_defined_categories():
    labels = np.zeros((4, 5), dtype=float)
    labels[:2, 0] = 1.0  # NORM: two positives, two negatives
    mask = np.ones((4, 5), dtype=bool)
    scores = np.zeros((4, 5), dtype=float)
    scores[:, 0] = [0.9, 0.8, 0.2, 0.1]
    thresholds, info = select_class_thresholds(labels, mask, scores)

    assert thresholds[0] == pytest.approx(0.8)
    assert info["categories"]["NORM"]["defined"] is True
    assert info["categories"]["NORM"]["validation_f1"] == pytest.approx(1.0)
    for category in TARGET_CATEGORIES[1:]:
        assert thresholds[list(TARGET_CATEGORIES).index(category)] == 0.5
        assert info["categories"][category]["defined"] is False
        assert info["categories"][category]["note"]
    assert info["tuned_on"] == "validation fold 9 only"
