"""Flask homepage and unified API tests (legacy routes preserved)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

import web.app as web_app
from src.ptbxl_data import LEAD_ORDER, TARGET_CATEGORIES
from src.unified_inference import UnifiedECGInference


def make_ecg(length: int = 1000, seed: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(len(LEAD_ORDER), length).astype(np.float32)


@pytest.fixture
def service(synthetic_artifact):
    return UnifiedECGInference(
        synthetic_artifact["version_dir"], device="cpu", load_now=True
    )


@pytest.fixture
def client(service, monkeypatch):
    monkeypatch.setattr(web_app, "_unified_service", lambda: (service, None))
    return web_app.app.test_client()


@pytest.fixture
def unavailable_client(monkeypatch):
    monkeypatch.setattr(
        web_app, "_unified_service", lambda: (None, "No unified artifact for this test")
    )
    return web_app.app.test_client()


def test_homepage_immediately_contains_required_sections():
    response = web_app.app.test_client().get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "HeartBeat Sentinel" in html
    assert "Integrated ECG Anomaly Detection and Heart Condition Classification" in html
    assert "Understand the Heart Conditions" in html
    assert "Analyze Your ECG" in html
    assert "Anomaly detection" in html
    assert "Condition classification" in html

    # Five exact condition cards, one per fixed target category.
    assert html.count('class="condition-card"') == 5
    for category in TARGET_CATEGORIES:
        assert f'data-category="{category}"' in html
    for lead in LEAD_ORDER:
        assert f'value="{lead}"' in html
    assert "Not for clinical use" in html
    assert "Anomaly detection and condition classification are separate outputs" not in html


def test_conditions_endpoint_has_exact_order():
    response = web_app.app.test_client().get("/api/unified/conditions")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["label_order"] == list(TARGET_CATEGORIES)
    assert [item["key"] for item in payload["conditions"]] == list(TARGET_CATEGORIES)
    assert "not a medical device" in payload["disclaimer"].lower()


def test_health_reports_legacy_and_unified_status():
    response = web_app.app.test_client().get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["task"] == "ecg5000_anomaly"
    assert "unified" in payload


def test_unified_health_unavailable_is_explicit(unavailable_client):
    response = unavailable_client.get("/api/unified/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["available"] is False
    assert "No unified artifact" in payload["error"]
    assert payload["expected_input"]["shape"] == [12, 1000]


def test_unified_predict_unavailable_returns_503(unavailable_client):
    response = unavailable_client.post(
        "/api/unified/predict",
        json={"signal": make_ecg().tolist(), "sampling_rate": 100},
    )
    assert response.status_code == 503
    payload = response.get_json()
    assert payload["type"] == "unified_model_unavailable"


def test_unified_predict_success_keeps_outputs_separate(client):
    response = client.post(
        "/api/unified/predict",
        json={"signal": make_ecg(seed=4).tolist(), "sampling_rate": 100},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["task"] == "unified_ptbxl"
    assert "anomaly_score" in payload
    assert "category_scores" in payload
    assert [item["category"] for item in payload["conditions"]["items"]] == list(
        TARGET_CATEGORIES
    )
    assert payload["anomaly"]["definition"]
    assert "reconstruction" in payload


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"signal": [1, 2, 3]},
        {"signal": make_ecg().tolist(), "sampling_rate": "not-a-number"},
        {"signal": np.zeros(140).tolist()},
    ],
)
def test_unified_predict_rejects_bad_payloads(client, payload):
    response = client.post("/api/unified/predict", json=payload)
    assert response.status_code == 400
    assert response.get_json()["type"] in {"invalid_request", "invalid_ecg_input"}


def test_unified_predict_invalid_json_returns_400(client):
    response = client.post(
        "/api/unified/predict", data="{not json", content_type="application/json"
    )
    assert response.status_code == 400
    assert response.get_json()["type"] == "invalid_request"


def test_unified_predict_file_npy(client):
    buffer = io.BytesIO()
    np.save(buffer, make_ecg(seed=5))
    buffer.seek(0)
    response = client.post(
        "/api/unified/predict-file",
        data={"file": (buffer, "ecg.npy")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["task"] == "unified_ptbxl"


def test_unified_predict_file_unsupported_type(client):
    response = client.post(
        "/api/unified/predict-file",
        data={"file": (io.BytesIO(b"nope"), "ecg.mat")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["type"] == "invalid_ecg_input"


def test_unified_predict_file_140_samples_is_400_not_500(client):
    buffer = io.BytesIO()
    np.save(buffer, np.zeros(140, dtype=np.float32))
    buffer.seek(0)
    response = client.post(
        "/api/unified/predict-file",
        data={"file": (buffer, "ecg.npy")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["type"] == "invalid_ecg_input"


def test_unified_explain_success_and_validation(client):
    response = client.post(
        "/api/unified/explain",
        json={"signal": make_ecg().tolist(), "category": "STTC"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["category"] == "STTC"
    assert len(payload["saliency"]) == 1000
    assert payload["method"] == "temporal_occlusion"

    bad = client.post(
        "/api/unified/explain",
        json={"signal": make_ecg().tolist(), "category": "NOPE"},
    )
    assert bad.status_code == 400
    assert bad.get_json()["type"] == "invalid_ecg_input"

    missing = client.post("/api/unified/explain", json={"signal": []})
    assert missing.status_code == 400
    assert missing.get_json()["type"] == "invalid_request"


def test_unified_examples_without_dataset_is_empty(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(web_app, "PTBXL_DATA_DIR", str(tmp_path / "no-ptbxl"))
    response = client.get("/api/unified/examples")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["examples"] == []
    assert "not found" in payload["message"]


def test_unified_example_without_dataset_is_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(web_app, "PTBXL_DATA_DIR", str(tmp_path / "no-ptbxl"))
    response = client.get("/api/unified/example/12345")
    assert response.status_code == 404
    assert response.get_json()["type"] == "ptbxl_data_error"


def test_unified_metrics_reports_missing_and_present_evaluation(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        web_app,
        "summary_or_error",
        lambda: {"available": False, "error": "no artifact"},
    )
    missing = client.get("/api/unified/metrics")
    assert missing.status_code == 200
    assert missing.get_json()["available"] is False

    evaluation = {
        "split": "val",
        "thresholds_frozen": True,
        "anomaly": {"threshold": 0.5, "n_evaluated": 2},
        "classification": {"per_category": {}, "macro_f1": None},
    }
    (tmp_path / "evaluation_val.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    monkeypatch.setattr(
        web_app,
        "summary_or_error",
        lambda: {"available": True, "version_dir": str(tmp_path)},
    )
    present = client.get("/api/unified/metrics")
    assert present.status_code == 200
    payload = present.get_json()
    assert payload["available"] is True
    assert payload["evaluations"]["val"]["split"] == "val"


def test_legacy_predict_endpoint_still_validates_140_values():
    client = web_app.app.test_client()
    response = client.post("/api/predict", json={"signal": [1, 2, 3]})
    assert response.status_code == 400
    payload = response.get_json()
    assert "error" in payload


def test_oversized_request_returns_structured_error(monkeypatch):
    monkeypatch.setitem(web_app.app.config, "MAX_CONTENT_LENGTH", 16)
    response = web_app.app.test_client().post(
        "/api/unified/predict", data=b"x" * 32, content_type="application/json"
    )
    assert response.status_code == 413
    assert response.get_json()["type"] == "request_too_large"
