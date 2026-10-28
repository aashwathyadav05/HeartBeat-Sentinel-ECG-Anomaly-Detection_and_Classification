import json
import logging
import os
import tempfile
from typing import Any, Dict, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request

from src.ptbxl_data import (
    LEAD_ORDER,
    TARGET_CATEGORIES,
    PTBXLDataError,
)
from src.ptbxl_examples import list_test_examples, load_test_record
from src.unified_inference import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_NAMES,
    DISCLAIMER,
    InvalidECGInputError,
    UnifiedInferenceError,
    load_signal_file,
    summary_or_error,
    try_get_unified_service,
)
from web.inference import inference_engine

LOGGER = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(
        {
            "error": "Request body exceeds the 25 MiB upload limit.",
            "type": "request_too_large",
        }
    ), 413

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PTBXL_DATA_DIR = os.environ.get("HEARTBEAT_PTBXL_DIR", os.path.join(PROJECT_ROOT, "data", "ptbxl"))
PTBXL_CACHE_DIR = os.environ.get("HEARTBEAT_PTBXL_CACHE", os.path.join(PROJECT_ROOT, "data", "ptbxl_cache"))

CONDITIONS = [
    {
        "key": category,
        "name": CATEGORY_NAMES[category],
        "description": CATEGORY_DESCRIPTIONS[category],
    }
    for category in TARGET_CATEGORIES
]


def _unified_service():
    return try_get_unified_service()


def _unified_error_response(error: str, status: int = 503):
    return (
        jsonify(
            {
                "error": error,
                "type": "unified_model_unavailable",
                "hint": (
                    "Train the unified PTB-XL model and tune thresholds, then verify "
                    "/api/unified/health. The ECG5000 workflow remains available separately."
                ),
            }
        ),
        status,
    )


# ---------------------------------------------------------------------------
# Existing ECG5000 routes (response formats preserved)
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template(
        "index.html",
        conditions=CONDITIONS,
        target_order=list(TARGET_CATEGORIES),
        lead_order=list(LEAD_ORDER),
        disclaimer=DISCLAIMER,
    )


@app.route("/api/health")
def health():
    legacy = inference_engine.health()
    return jsonify(
        {
            "status": "ok",
            "model_loaded": inference_engine.model is not None,
            "threshold": float(inference_engine.threshold) if inference_engine.threshold is not None else None,
            "seq_len": inference_engine.seq_len,
            "task": "ecg5000_anomaly",
            "error": inference_engine.load_error,
            "unified": summary_or_error(),
        }
    )


@app.route("/api/metrics")
def metrics():
    try:
        with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as handle:
            metric_payload = json.load(handle)
        meta = {}
        metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        return jsonify({"metrics": metric_payload, "metadata": meta})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/examples")
def examples():
    try:
        X_test = np.load(os.path.join(DATA_DIR, "test_data.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
        items = []
        for index in range(min(20, len(X_test))):
            items.append({"index": index, "true_label": float(y_test[index])})
        return jsonify({"examples": items, "count": len(X_test)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/example/<int:idx>")
def example(idx):
    try:
        X_test = np.load(os.path.join(DATA_DIR, "test_data.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
        if idx < 0 or idx >= len(X_test):
            return jsonify({"error": "index out of range"}), 400
        res = inference_engine.predict(X_test[idx])
        res["true_label"] = float(y_test[idx])
        res["index"] = idx
        return jsonify(res)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    if "signal" not in data:
        return jsonify({"error": "Missing 'signal' field"}), 400
    signal = data["signal"]
    if not isinstance(signal, (list, tuple)):
        return jsonify({"error": "'signal' must be a list"}), 400
    if len(signal) != inference_engine.seq_len:
        return jsonify({"error": f"Signal must have exactly {inference_engine.seq_len} elements"}), 400
    import math

    for index, value in enumerate(signal):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return jsonify({"error": f"Non-numeric value at index {index}"}), 400
        if not math.isfinite(value):
            return jsonify({"error": f"Non-finite value at index {index}"}), 400
    if inference_engine.model is None:
        return (
            jsonify(
                {
                    "error": inference_engine.load_error or "ECG5000 model is not loaded",
                    "hint": "Train the legacy model with 'python src/train.py'.",
                    "task": "ecg5000_anomaly",
                }
            ),
            503,
        )
    try:
        res = inference_engine.predict(signal)
        if "true_label" in data:
            res["true_label"] = data["true_label"]
        return jsonify(res)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        LOGGER.exception("Legacy prediction failed")
        return jsonify({"error": "Prediction failed"}), 500


# ---------------------------------------------------------------------------
# Unified PTB-XL routes
# ---------------------------------------------------------------------------


@app.route("/api/unified/health")
def unified_health():
    service, error = _unified_service()
    if service is None:
        return (
            jsonify(
                {
                    "available": False,
                    "error": error,
                    "expected_input": {
                        "shape": [12, 1000],
                        "lead_order": list(LEAD_ORDER),
                        "sampling_rate": 100,
                    },
                    "legacy_ecg5000_available": inference_engine.model is not None,
                }
            ),
            200,
        )
    payload = service.health()
    payload["available"] = True
    payload["legacy_ecg5000_available"] = inference_engine.model is not None
    return jsonify(payload)


@app.route("/api/unified/conditions")
def unified_conditions():
    return jsonify(
        {
            "label_order": list(TARGET_CATEGORIES),
            "conditions": CONDITIONS,
            "disclaimer": DISCLAIMER,
        }
    )


@app.route("/api/unified/metrics")
def unified_metrics():
    payload: Dict[str, Any] = {"available": False, "evaluations": {}}
    try:
        artifact_summary = summary_or_error()
        payload["artifact"] = artifact_summary
        if artifact_summary.get("available"):
            version_dir = artifact_summary["version_dir"]
            for split in ("val", "test"):
                path = os.path.join(version_dir, f"evaluation_{split}.json")
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as handle:
                        payload["evaluations"][split] = json.load(handle)
            payload["available"] = True
        else:
            payload["error"] = artifact_summary.get("error")
    except Exception as exc:
        payload["error"] = str(exc)
    return jsonify(payload)


def _list_examples_or_error():
    if not os.path.isdir(PTBXL_DATA_DIR):
        return None, f"PTB-XL data directory not found: {PTBXL_DATA_DIR}"
    try:
        return list_test_examples(PTBXL_DATA_DIR), None
    except PTBXLDataError as exc:
        return None, str(exc)


def _read_test_record(ecg_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    return load_test_record(
        PTBXL_DATA_DIR,
        ecg_id,
        cache_dir=PTBXL_CACHE_DIR if os.path.isdir(PTBXL_CACHE_DIR) else None,
    )


@app.route("/api/unified/examples")
def unified_examples():
    items, error = _list_examples_or_error()
    if items is None:
        return jsonify({"examples": [], "count": 0, "message": error}), 200
    return jsonify(
        {
            "examples": items,
            "count": len(items),
            "message": "PTB-XL test records. Dataset labels are shown for evaluation context only.",
        }
    )


@app.route("/api/unified/example/<int:ecg_id>")
def unified_example(ecg_id):
    service, error = _unified_service()
    if service is None:
        return _unified_error_response(error or "Unified model unavailable")
    try:
        signal, info = _read_test_record(ecg_id)
        result = service.analyze(signal, source=f"ptbxl_test_record:{ecg_id}")
        result["record"] = info
        return jsonify(result)
    except InvalidECGInputError as exc:
        return jsonify({"error": str(exc), "type": "invalid_ecg_input"}), 400
    except PTBXLDataError as exc:
        return jsonify({"error": str(exc), "type": "ptbxl_data_error"}), 404
    except Exception as exc:
        LOGGER.exception("Unified example prediction failed")
        return jsonify({"error": str(exc), "type": "inference_error"}), 500


def _predict_payload(data: Dict[str, Any]):
    if "signal" not in data:
        return jsonify({"error": "Missing 'signal' field (12 rows x 1000 samples)", "type": "invalid_ecg_input"}), 400
    signal = data["signal"]
    sampling_rate = data.get("sampling_rate", 100)
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError):
        return jsonify({"error": "sampling_rate must be numeric", "type": "invalid_ecg_input"}), 400
    service, error = _unified_service()
    if service is None:
        return _unified_error_response(error or "Unified model unavailable")
    try:
        result = service.analyze(signal, sampling_rate=sampling_rate)
        return jsonify(result)
    except InvalidECGInputError as exc:
        return jsonify({"error": str(exc), "type": "invalid_ecg_input"}), 400
    except UnifiedInferenceError as exc:
        return _unified_error_response(str(exc))
    except Exception as exc:
        LOGGER.exception("Unified prediction failed")
        return jsonify({"error": str(exc), "type": "inference_error"}), 500


@app.route("/api/unified/predict", methods=["POST"])
def unified_predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON", "type": "invalid_request"}), 400
    return _predict_payload(data)


@app.route("/api/unified/predict-file", methods=["POST"])
def unified_predict_file():
    if "file" not in request.files:
        return jsonify({"error": "Missing multipart 'file' field", "type": "invalid_request"}), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Uploaded file has no filename", "type": "invalid_request"}), 400
    suffix = os.path.splitext(upload.filename)[1].lower()
    if suffix not in {".npy", ".csv", ".txt", ".json"}:
        return jsonify(
            {
                "error": "Unsupported file type. Upload .npy, .csv, .txt or .json with one 12x1000 ECG.",
                "type": "invalid_ecg_input",
            }
        ), 400
    service, error = _unified_service()
    if service is None:
        return _unified_error_response(error or "Unified model unavailable")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        upload.save(handle.name)
        handle.close()
        signal = load_signal_file(handle.name, filename=upload.filename)
        result = service.analyze(signal, source=f"upload:{upload.filename}")
        return jsonify(result)
    except InvalidECGInputError as exc:
        return jsonify({"error": str(exc), "type": "invalid_ecg_input"}), 400
    except UnifiedInferenceError as exc:
        return _unified_error_response(str(exc))
    except Exception as exc:
        LOGGER.exception("Unified file prediction failed")
        return jsonify({"error": str(exc), "type": "inference_error"}), 500
    finally:
        try:
            handle.close()
        except Exception:
            pass
        try:
            os.remove(handle.name)
        except OSError:
            pass


@app.route("/api/unified/explain", methods=["POST"])
def unified_explain():
    data = request.get_json(force=True, silent=True)
    if not data or "signal" not in data or "category" not in data:
        return jsonify(
            {"error": "JSON must include 'signal' and 'category'", "type": "invalid_request"}
        ), 400
    service, error = _unified_service()
    if service is None:
        return _unified_error_response(error or "Unified model unavailable")
    try:
        result = service.explain(
            data["signal"],
            str(data["category"]),
            sampling_rate=float(data.get("sampling_rate", 100)),
            window_samples=int(data.get("window_samples", 100)),
            stride_samples=int(data.get("stride_samples", 50)),
        )
        return jsonify(result)
    except InvalidECGInputError as exc:
        return jsonify({"error": str(exc), "type": "invalid_ecg_input"}), 400
    except UnifiedInferenceError as exc:
        return _unified_error_response(str(exc))
    except Exception as exc:
        LOGGER.exception("Unified explanation failed")
        return jsonify({"error": str(exc), "type": "explanation_error"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
