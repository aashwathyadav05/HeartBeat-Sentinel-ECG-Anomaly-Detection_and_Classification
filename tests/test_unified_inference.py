"""Unified inference service response contract and validation tests."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.ptbxl_data import LEAD_ORDER, TARGET_CATEGORIES, normalize_signal
from src.unified_inference import (
    InvalidECGInputError,
    ModelNotAvailableError,
    UnifiedECGInference,
    load_signal_file,
)


def make_ecg(length: int = 1000, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(len(LEAD_ORDER), length).astype(np.float32)


@pytest.fixture
def service(synthetic_artifact):
    return UnifiedECGInference(
        synthetic_artifact["version_dir"], device="cpu", load_now=True
    )


def test_health_contract(service):
    health = service.health()
    assert health["available"] is True
    assert health["model_version"]
    assert health["label_order"] == list(TARGET_CATEGORIES)
    assert health["expected_input"]["shape"] == [12, 1000]
    assert health["expected_input"]["lead_order"] == list(LEAD_ORDER)
    assert health["expected_input"]["sampling_rate"] == 100
    assert health["expected_input"]["normalization"] == "train_lead_zscore"
    assert health["anomaly_threshold"] == 0.5
    assert health["class_thresholds"] == [0.5] * 5


def test_analyze_separates_anomaly_and_condition_outputs(service, synthetic_artifact):
    signal = make_ecg(seed=11)
    response = service.analyze(signal)

    assert response["task"] == "unified_ptbxl"
    assert response["anomaly_threshold"] == 0.5
    assert response["anomaly"]["threshold"] == 0.5
    assert response["anomaly"]["per_lead_scores"].__len__() == 12
    assert response["anomaly"]["lead_order"] == list(LEAD_ORDER)
    assert response["anomaly_status"] == (
        "anomalous" if response["anomaly_score"] > response["anomaly_threshold"] else "normal"
    )
    assert response["anomaly"]["status"] == response["anomaly_status"]

    assert list(response["category_scores"]) == list(TARGET_CATEGORIES)
    assert response["conditions"]["scores_are_independent"] is True
    assert [item["category"] for item in response["conditions"]["items"]] == list(
        TARGET_CATEGORIES
    )
    for item in response["conditions"]["items"]:
        assert 0.0 <= item["score"] <= 1.0
        assert item["predicted"] == (item["score"] >= item["threshold"])
        assert "not a calibrated clinical probability" in item["score_type"]
    assert response["predicted_categories"] == [
        item["category"] for item in response["conditions"]["items"] if item["predicted"]
    ]

    reconstruction = response["reconstruction"]
    assert np.asarray(reconstruction["original"]).shape == (12, 1000)
    assert np.asarray(reconstruction["normalized_input"]).shape == (12, 1000)
    assert np.asarray(reconstruction["reconstructed"]).shape == (12, 1000)
    assert np.asarray(reconstruction["pointwise_squared_error"]).shape == (12, 1000)
    expected_normalized = normalize_signal(
        np.asarray(signal), synthetic_artifact["normalization"]
    )
    assert np.allclose(reconstruction["normalized_input"], expected_normalized, atol=1e-6)

    # The response must label the anomaly as a separate, non-diagnostic task.
    assert response["metadata"]["label_order"] == list(TARGET_CATEGORIES)
    assert "not a medical device" in response["disclaimer"].lower()
    assert any("separate outputs" in note for note in response["notes"])


def test_analyze_can_omit_large_signal_payload(service):
    response = service.analyze(make_ecg(), include_signal=False)
    assert "reconstruction" not in response
    assert response["anomaly_score"] >= 0.0


def test_analyze_rejects_140_sample_and_bad_rate(service):
    with pytest.raises(InvalidECGInputError, match="ECG5000"):
        service.analyze(np.zeros(140, dtype=np.float32))
    with pytest.raises(InvalidECGInputError, match="100 Hz"):
        service.analyze(make_ecg(), sampling_rate=500)
    with pytest.raises(InvalidECGInputError, match="12 leads"):
        service.analyze(np.zeros((11, 1000), dtype=np.float32))


def test_occlusion_explanation_contract(service):
    result = service.explain(make_ecg(), "MI", window_samples=100, stride_samples=50)
    assert result["method"] == "temporal_occlusion"
    assert result["category"] == "MI"
    assert len(result["saliency"]) == 1000
    assert result["window_samples"] == 100
    assert result["stride_samples"] == 50
    assert result["regions"]
    assert len(result["top_regions"]) <= 5
    assert "not evidence of causal physiology" in result["disclaimer"]
    with pytest.raises(InvalidECGInputError, match="category"):
        service.explain(make_ecg(), "NOT_A_CATEGORY")


def test_analyze_file_formats(service, tmp_path):
    signal = make_ecg(seed=21)

    npy_path = tmp_path / "ecg.npy"
    np.save(npy_path, signal)
    assert service.analyze_file(str(npy_path))["reconstruction"]["original"]

    json_path = tmp_path / "ecg.json"
    json_path.write_text(json.dumps({"signal": signal.tolist()}), encoding="utf-8")
    assert service.analyze_file(str(json_path))["anomaly_score"] >= 0.0

    csv_path = tmp_path / "ecg.csv"
    np.savetxt(csv_path, signal, delimiter=",")
    assert service.analyze_file(str(csv_path))["anomaly_score"] >= 0.0

    text_path = tmp_path / "ecg.txt"
    np.savetxt(text_path, signal)
    assert service.analyze_file(str(text_path))["anomaly_score"] >= 0.0

    bad_path = tmp_path / "ecg.npy"
    np.save(bad_path, np.zeros(140, dtype=np.float32))
    with pytest.raises(InvalidECGInputError, match="ECG5000"):
        service.analyze_file(str(bad_path))

    unsupported = tmp_path / "ecg.mat"
    unsupported.write_text("not an ECG", encoding="utf-8")
    with pytest.raises(InvalidECGInputError, match="Unsupported"):
        service.analyze_file(str(unsupported))


def test_load_signal_file_rejects_malformed_payloads(tmp_path):
    missing_signal = tmp_path / "bad.json"
    missing_signal.write_text(json.dumps({"not_signal": []}), encoding="utf-8")
    with pytest.raises(InvalidECGInputError, match="signal"):
        load_signal_file(str(missing_signal))

    malformed = tmp_path / "bad.npy"
    malformed.write_text("not numpy", encoding="utf-8")
    with pytest.raises(InvalidECGInputError):
        load_signal_file(str(malformed))


def test_missing_artifact_is_reported_without_crashing(tmp_path):
    missing = str(tmp_path / "does-not-exist")
    service = UnifiedECGInference(missing, load_now=False)
    assert service.is_available is False
    with pytest.raises(ModelNotAvailableError):
        service.require_loaded()
    health = service.health()
    assert health["available"] is False
    assert health["expected_input"]["shape"] == [12, 1000]

    with pytest.raises(ModelNotAvailableError):
        UnifiedECGInference(missing, load_now=True)
