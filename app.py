"""HeartBeat Sentinel Streamlit dashboard.

Both workflows call the same services as the Flask app:
``src.unified_inference`` for PTB-XL and ``src.ecg5000_inference`` for the
legacy 140-sample autoencoder. No model inference logic is reimplemented here.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.ecg5000_inference import try_load_legacy
from src.ptbxl_data import LEAD_ORDER, TARGET_CATEGORIES
from src.ptbxl_examples import list_test_examples, load_test_record
from src.unified_artifacts import artifact_paths, resolve_artifact_dir
from src.unified_explain import occlusion_sensitivity
from src.unified_inference import (
    DISCLAIMER,
    InvalidECGInputError,
    UnifiedInferenceError,
    load_signal_file,
    try_get_unified_service,
)
from src.unified_visualize import (
    plot_condition_scores,
    plot_occlusion,
    plot_reconstruction,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PTBXL_DIR = os.environ.get("HEARTBEAT_PTBXL_DIR", os.path.join(PROJECT_ROOT, "data", "ptbxl"))
PTBXL_CACHE = os.environ.get("HEARTBEAT_PTBXL_CACHE", os.path.join(PROJECT_ROOT, "data", "ptbxl_cache"))

st.set_page_config(page_title="HeartBeat Sentinel", layout="wide")


@st.cache_resource
def load_services():
    unified, unified_error = try_get_unified_service()
    legacy, legacy_error = try_load_legacy()
    return unified, unified_error, legacy, legacy_error


def show_figure(figure) -> None:
    if figure is None:
        return
    st.pyplot(figure)
    plt.close(figure)


def short_error(exc: Exception) -> str:
    return str(exc)


def render_unified_result(service, result) -> None:
    st.subheader("Anomaly detection")
    st.caption(
        "Mean squared reconstruction error in normalized units. It measures deviation "
        "from learned normal reconstruction and does not identify a condition."
    )
    first, second, third = st.columns(3)
    first.metric("Reconstruction error", f"{result['anomaly_score']:.6f}")
    second.metric("Frozen threshold", f"{result['anomaly_threshold']:.6f}")
    third.metric(
        "Anomaly threshold",
        "Above threshold" if result["anomaly_status"] == "anomalous" else "Below threshold",
    )

    st.subheader("Condition classification")
    st.caption(
        "Independent sigmoid scores compared with five validation-tuned thresholds. "
        "A score is not a calibrated clinical probability."
    )
    items = result["conditions"]["items"]
    flagged = [item["category"] for item in items if item["predicted"]]
    if "NORM" in flagged and any(category != "NORM" for category in flagged):
        st.warning(
            "Conflicting model flags: NORM and an abnormal category are both above threshold. "
            "This can be a classifier false positive; neither output is a confirmed diagnosis."
        )
    for item in items:
        label = f"{item['category']} — {item['name']}"
        state = "above threshold" if item["predicted"] else "below threshold"
        st.markdown(
            f"**{label}** · score `{item['score']:.4f}` · threshold "
            f"`{item['threshold']:.4f}` · **{state}**"
        )
        st.progress(float(max(0.0, min(1.0, item["score"]))))
    predicted = result.get("predicted_categories") or []
    if predicted:
        st.warning(
            "Flagged categories: " + ", ".join(predicted) +
            ". These are model outputs, not diagnoses. An anomaly is not evidence of these conditions."
        )
    else:
        st.info(
            "No category reached its frozen threshold. This does not rule out every "
            "possible heart condition."
        )

    reconstruction = result.get("reconstruction", {})
    if reconstruction:
        st.subheader("Reconstruction and error")
        lead = st.selectbox("Lead", LEAD_ORDER, index=LEAD_ORDER.index("II"))
        figure = plot_reconstruction(
            np.asarray(reconstruction["normalized_input"]),
            np.asarray(reconstruction["reconstructed"]),
            np.asarray(reconstruction["pointwise_squared_error"]),
            lead=lead,
            anomaly_score=result["anomaly_score"],
            anomaly_threshold=result["anomaly_threshold"],
        )
        show_figure(figure)
        score_figure = plot_condition_scores(
            result["category_scores"], result["category_thresholds"]
        )
        show_figure(score_figure)

        st.subheader("Model sensitivity (optional)")
        st.caption(
            "Temporal occlusion measures how replacing short windows with the training "
            "mean changes one category logit. It is not a causal or clinical explanation."
        )
        category = st.selectbox("Category to probe", list(TARGET_CATEGORIES))
        if st.button("Compute occlusion sensitivity", key="unified_explain"):
            try:
                explanation = service.explain(
                    reconstruction["original"], category, window_samples=100, stride_samples=50
                )
                figure = plot_occlusion(
                    explanation["saliency"],
                    category=category,
                    window_samples=explanation["window_samples"],
                )
                show_figure(figure)
                st.caption(explanation["disclaimer"])
            except (InvalidECGInputError, UnifiedInferenceError) as exc:
                st.error(short_error(exc))
            except Exception as exc:
                st.error(f"Explanation failed: {exc}")

    with st.expander("Response metadata"):
        st.json(
            {
                "model_version": result["metadata"].get("model_version"),
                "artifact_dir": result["metadata"].get("artifact_dir"),
                "dataset": {
                    "name": result["metadata"].get("dataset", {}).get("name"),
                    "version": result["metadata"].get("dataset", {}).get("version"),
                    "sampling_rate": result["metadata"].get("dataset", {}).get("sampling_rate"),
                },
                "input_shape": result["metadata"].get("input_shape"),
                "lead_order": result["metadata"].get("input_lead_order"),
            }
        )
    st.caption(DISCLAIMER)


def unified_tab(service, unified_error: str | None) -> None:
    st.header("Analyze Your ECG — Unified PTB-XL Model")
    if service is None:
        st.error("The unified PTB-XL model is not available.")
        st.markdown(
            "Train the model and tune its validation thresholds first:\n"
            "```\npython -m src.train_unified_model all --data-dir data/ptbxl\n```\n"
            "Then restart or reload this dashboard. The legacy ECG5000 workflow is still "
            "available in its own tab."
        )
        if unified_error:
            st.code(unified_error)
        return

    health = service.health()
    st.caption(
        f"Model {health.get('model_version') or 'ready'} · 12 leads · 100 Hz · 10 seconds"
    )

    method = st.radio(
        "Input method",
        ("Upload ECG file", "PTB-XL test record", "Paste JSON"),
        horizontal=True,
    )
    signal = None
    info = None
    if method == "Upload ECG file":
        upload = st.file_uploader("Upload a 12×1000 ECG", type=["npy", "csv", "txt", "json"])
        if upload is not None:
            suffix = os.path.splitext(upload.name)[1].lower()
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                handle.write(upload.getvalue())
                handle.close()
                signal = load_signal_file(handle.name, filename=upload.name)
            except (InvalidECGInputError, Exception) as exc:
                st.error(f"Invalid ECG file: {exc}")
            finally:
                try:
                    handle.close()
                except Exception:
                    pass
                try:
                    os.remove(handle.name)
                except OSError:
                    pass
    elif method == "PTB-XL test record":
        try:
            examples = list_test_examples(PTBXL_DIR, limit=20)
        except Exception as exc:
            examples = []
            st.warning(f"PTB-XL examples are unavailable: {exc}")
        if examples:
            labels = {
                item["ecg_id"]: ", ".join(item["true_categories"]) or "no mapped label"
                for item in examples
            }
            ecg_id = st.selectbox(
                "Held-out test record (dataset label shown for evaluation context only)",
                list(labels.keys()),
                format_func=lambda value: f"ecg_id {value} — {labels[value]}",
            )
            try:
                signal, info = load_test_record(
                    PTBXL_DIR,
                    int(ecg_id),
                    cache_dir=PTBXL_CACHE if os.path.isdir(PTBXL_CACHE) else None,
                )
                if info:
                    st.caption(
                        f"Dataset labels: {', '.join(info['true_categories']) or 'none'} · "
                        f"fold {info['fold']} · {info['label_note']}"
                    )
            except Exception as exc:
                st.error(f"Could not load test record: {exc}")
    else:
        pasted = st.text_area(
            "Paste one 12×1000 JSON array (list of 12 rows, or an object with a 'signal' field)",
            height=180,
        )
        if pasted.strip():
            try:
                data = json.loads(pasted)
                if isinstance(data, dict) and "signal" in data:
                    data = data["signal"]
                signal = np.asarray(data)
            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")

    if st.button("Run integrated analysis", type="primary", disabled=signal is None):
        try:
            with st.spinner("Running shared encoder, reconstruction and classifier…"):
                result = service.analyze(signal, source=f"streamlit:{method}")
            st.session_state["unified_result"] = result
        except InvalidECGInputError as exc:
            st.error(f"Invalid ECG input: {exc}")
        except UnifiedInferenceError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("unified_result")
    if result:
        render_unified_result(service, result)


def legacy_tab(legacy, legacy_error: str | None) -> None:
    st.header("Legacy ECG5000 Anomaly Model")
    st.caption(
        "Original 140-sample convolutional autoencoder. This workflow is intentionally "
        "separate from the 12-lead PTB-XL model."
    )
    if legacy is None or legacy.model is None:
        st.error("The legacy ECG5000 model is not available.")
        if legacy_error or (legacy and legacy.load_error):
            st.code(legacy_error or legacy.load_error)
        st.markdown("Train it with:\n```\npython src/train.py\n```")
        return

    signal = None
    source = st.radio("Input method", ("Dataset test segment", "Paste 140 values"), horizontal=True)
    if source == "Dataset test segment":
        data_path = os.path.join(DATA_DIR, "test_data.npy")
        labels_path = os.path.join(DATA_DIR, "test_labels.npy")
        if os.path.exists(data_path):
            X_test = np.load(data_path)
            y_test = np.load(labels_path) if os.path.exists(labels_path) else None
            index = st.slider("Test segment index", 0, len(X_test) - 1, 0)
            signal = X_test[index]
            if y_test is not None:
                st.caption(f"Dataset label: {y_test[index]} (evaluation context only)")
        else:
            st.info("No ECG5000 test arrays found. Run the preprocessing step or paste 140 values.")
    else:
        pasted = st.text_area("Paste exactly 140 comma- or space-separated values", height=130)
        if pasted.strip():
            try:
                signal = np.asarray([float(value) for value in pasted.replace("\n", " ").replace(",", " ").split()])
            except ValueError:
                st.error("All values must be numeric.")

    if st.button("Run ECG5000 anomaly detection", type="primary", disabled=signal is None):
        try:
            result = legacy.predict(signal)
            st.session_state["legacy_result"] = result
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("legacy_result")
    if result:
        columns = st.columns(3)
        columns[0].metric("Reconstruction MSE", f"{result['segment_mse']:.6f}")
        columns[1].metric("Threshold", f"{result['threshold']:.6f}")
        columns[2].metric("Status", result["predicted_label"].upper())
        figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(result["original"], label="Original", color="#3b82f6")
        axes[0].plot(result["reconstructed"], label="Reconstructed", color="#10b981")
        axes[0].set_title("ECG5000 original vs reconstruction")
        axes[0].set_ylabel("Dataset units")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(result["pointwise_squared_error"], color="#ef4444")
        axes[1].fill_between(
            range(len(result["pointwise_squared_error"])),
            result["pointwise_squared_error"],
            color="#ef4444",
            alpha=0.25,
        )
        axes[1].set_title("Point-wise squared reconstruction error")
        axes[1].set_xlabel("Sample index")
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        show_figure(figure)
        st.caption(
            "Reconstruction error flags deviation from learned normal segments. "
            "It is not a diagnosis."
        )


def evaluation_tab() -> None:
    st.header("Evaluation")
    st.caption("Anomaly and condition metrics are reported separately and are not compared.")
    try:
        version_dir = resolve_artifact_dir()
        found_any = False
        for split in ("val", "test"):
            path = os.path.join(version_dir, f"evaluation_{split}.json")
            if not os.path.exists(path):
                continue
            found_any = True
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            st.subheader(f"Unified PTB-XL — {split.upper()}")
            st.caption(
                f"Thresholds frozen: {result.get('thresholds_frozen')} · "
                f"anomaly threshold: {result['anomaly'].get('threshold')}"
            )
            anomaly = result["anomaly"]
            columns = st.columns(5)
            columns[0].metric("Precision", str(anomaly.get("precision")))
            columns[1].metric("Recall", str(anomaly.get("recall")))
            columns[2].metric("F1", str(anomaly.get("f1")))
            columns[3].metric("AUROC", str(anomaly.get("auroc")))
            columns[4].metric("Evaluated", str(anomaly.get("n_evaluated")))
            rows = []
            for category, values in result["classification"]["per_category"].items():
                rows.append(
                    {
                        "category": category,
                        "threshold": values["threshold"],
                        "support": values["positive_support"],
                        "predicted_positive": values["predicted_positive_count"],
                        "precision": values["precision"],
                        "recall": values["recall"],
                        "f1": values["f1"],
                        "auroc": values["auroc"],
                        "average_precision": values["average_precision"],
                    }
                )
            st.dataframe(rows, use_container_width=True)
            st.caption(
                f"Macro-F1: {result['classification'].get('macro_f1')} · "
                f"Micro-F1: {result['classification'].get('micro_f1')} · "
                "None/undefined metrics are shown explicitly."
            )
        if not found_any:
            st.info(
                "No unified evaluation files found. Run "
                "`python -m src.unified_evaluate --split val` and, once thresholds are "
                "frozen, `--split test`."
            )
    except Exception as exc:
        st.info(f"Unified evaluation unavailable: {exc}")

    st.subheader("Legacy ECG5000")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        columns = st.columns(5)
        columns[0].metric("Precision", f"{metrics['precision']:.3f}")
        columns[1].metric("Recall", f"{metrics['recall_sensitivity']:.3f}")
        columns[2].metric("F1", f"{metrics['f1_score']:.3f}")
        columns[3].metric("AUROC", f"{metrics['auroc']:.3f}")
        columns[4].metric("Threshold", f"{metrics['threshold']:.6f}")
        for name in ("confusion_matrix.png", "error_distribution.png", "loss_curves.png"):
            path = os.path.join(MODEL_DIR, name)
            if os.path.exists(path):
                st.image(path, caption=name.replace("_", " ").replace(".png", ""))
    else:
        st.info("No legacy ECG5000 metrics found.")


def limitations_tab() -> None:
    st.markdown(
        """
**Unified model training**

- Shared 1D CNN encoder → reconstruction decoder and five-logit classifier.
- Stage A: reconstruction pretraining on training-partition NORM-only records.
- Stage B: multilabel BCEWithLogitsLoss with unknown labels masked and training-only class weighting.
- Stage C: optional joint fine-tuning with reconstruction restricted to NORM-only records by an explicit mask.
- Official PTB-XL folds: 1–8 train, 9 validation, 10 test; patient leakage is checked.
- Anomaly threshold from validation-normal reconstruction errors; category thresholds tuned on validation labels and frozen before test.
- Input: 12 leads × 1000 samples at 100 Hz, fixed lead order, training-normalization reused at inference.

**Limitations**

- Educational prototype only; not a medical device.
- PTB-XL diagnostic superclasses are broad labels and do not cover every abnormality.
- No external validation, probability calibration or prospective clinical testing.
- Occlusion sensitivity is a model probe, not causal evidence.
- Anomaly and condition scores are on different scales and are not comparable.
- Do not use these outputs for diagnosis or treatment decisions.
        """
    )


def main() -> None:
    st.title("HeartBeat Sentinel")
    st.caption(
        "Integrated ECG Anomaly Detection and Heart Condition Classification · "
        "Educational research prototype — not for clinical use"
    )

    unified, unified_error, legacy, legacy_error = load_services()
    if st.sidebar.button("Reload model artifacts"):
        load_services.clear()
        st.rerun()
    tabs = st.tabs(["Analyze", "ECG5000", "Results & guide"])
    with tabs[0]:
        unified_tab(unified, unified_error)
    with tabs[1]:
        legacy_tab(legacy, legacy_error)
    with tabs[2]:
        evaluation_tab()
        with st.expander("How it works and limitations"):
            limitations_tab()


if __name__ == "__main__":
    main()
