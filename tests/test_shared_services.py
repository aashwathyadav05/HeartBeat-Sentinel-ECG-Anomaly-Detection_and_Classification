"""Both frontends must use the shared inference services, not duplicate them."""

from __future__ import annotations

import os
import re

import src.ecg5000_inference as ecg5000_service
import src.unified_inference as unified_service
import web.app as web_app
import web.inference as web_inference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_flask_legacy_module_reexports_shared_service():
    assert web_inference.ECG5000Inference is ecg5000_service.ECG5000Inference
    assert isinstance(web_inference.inference_engine, ecg5000_service.ECG5000Inference)
    assert web_app.inference_engine is web_inference.inference_engine


def test_flask_uses_shared_unified_service():
    assert web_app.try_get_unified_service is unified_service.try_get_unified_service
    assert web_app.summary_or_error is unified_service.summary_or_error


def test_streamlit_calls_both_shared_services():
    source = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    assert "from src.unified_inference import" in source
    assert "try_get_unified_service" in source
    assert "from src.ecg5000_inference import" in source
    assert "try_load_legacy" in source
    # The dashboard must call service methods rather than wrapping torch models.
    assert re.search(r"service\.analyze\(", source)
    assert re.search(r"legacy\.predict\(", source)
    assert "no model inference logic is reimplemented" in source.lower()


def test_flask_calls_shared_service_methods():
    source = open(os.path.join(ROOT, "web", "app.py"), encoding="utf-8").read()
    assert "service.analyze(" in source
    assert "service.explain(" in source
    assert "inference_engine.predict(" in source
    # No PTB-XL auto-download or auto-training at import/startup.
    assert "download" not in source.lower()
    assert "subprocess" not in source
    assert "fit_lead_normalization" not in source
