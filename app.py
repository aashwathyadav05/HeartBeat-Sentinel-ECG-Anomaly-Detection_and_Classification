import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st

from src.anomaly_detection import load_artifacts
from src.preprocess import prepare_dataloaders
from src.model import ConvAutoencoder

st.set_page_config(page_title="HeartBeat Sentinel", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_OUT = os.path.join(os.path.dirname(__file__), "data")


@st.cache_resource
def get_model():
    model, device, ckpt = load_artifacts()
    return model, device


@st.cache_data
def get_data():
    info = prepare_dataloaders()
    return info


def main():
    st.title("HeartBeat Sentinel: Explainable ECG Anomaly Detection")
    st.caption("Educational research prototype - not a clinical diagnostic system")

    tabs = st.tabs(["Overview", "ECG Analysis", "Evaluation", "Limitations"])

    with tabs[0]:
        st.header("Overview")
        st.markdown("""
        **Objective:** Learn normal ECG patterns with a 1D Convolutional Autoencoder and flag deviations via reconstruction error.

        **Dataset:** ECG5000 (UCR) - fixed-length 140-sample ECG segments. Autoencoder trained only on normal-labelled segments.

        **Architecture:** Conv1d encoder -> latent (FC) -> ConvTranspose1d decoder, MSE reconstruction loss.
        """)
        if os.path.exists(os.path.join(MODEL_DIR, "loss_curves.png")):
            st.image(os.path.join(MODEL_DIR, "loss_curves.png"), caption="Training/Validation Loss")

    with tabs[1]:
        st.header("ECG Analysis")
        try:
            info = get_data()
            X_test = info["X_test"]
            y_test = info["y_test"]
            model, device = get_model()
            idx = st.slider("Select test segment index", 0, len(X_test) - 1, 0)
            x = torch.from_numpy(X_test[idx:idx + 1]).unsqueeze(1).to(device)
            with torch.no_grad():
                x_hat = model(x)
            x_np = x.squeeze().cpu().numpy()
            xh_np = x_hat.squeeze().cpu().numpy()
            pt_err = (x_np - xh_np) ** 2
            seg_err = pt_err.mean()
            try:
                with open(os.path.join(MODEL_DIR, "threshold.json")) as f:
                    thresh = json.load(f)["threshold"]
            except Exception:
                thresh = seg_err
            flag = "ANOMALOUS" if seg_err > thresh else "NORMAL"
            c1, c2, c3 = st.columns(3)
            c1.metric("Segment MSE", f"{seg_err:.6f}")
            c2.metric("Threshold", f"{thresh:.6f}")
            c3.metric("Flag", flag)
            fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(x_np, label="Original")
            ax[0].plot(xh_np, label="Reconstructed")
            ax[0].legend()
            ax[0].set_title(f"True label: {y_test[idx]}")
            ax[1].plot(pt_err)
            ax[1].set_title("Point-wise Squared Error")
            ax[2].fill_between(range(len(pt_err)), pt_err, alpha=0.4)
            ax[2].set_title("Error Highlight")
            ax[2].set_xlabel("Time")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}. Make sure model is trained and files exist.")

    with tabs[2]:
        st.header("Evaluation")
        if os.path.exists(os.path.join(MODEL_DIR, "metrics.json")):
            with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
                m = json.load(f)
            cols = st.columns(6)
            cols[0].metric("Precision", f"{m['precision']:.3f}")
            cols[1].metric("Recall/Sens", f"{m['recall_sensitivity']:.3f}")
            cols[2].metric("Specificity", f"{m['specificity']:.3f}")
            cols[3].metric("F1", f"{m['f1_score']:.3f}")
            cols[4].metric("AUROC", f"{m['auroc']:.3f}")
            cols[5].metric("Threshold", f"{m['threshold']:.6f}")
            st.json(m)
        if os.path.exists(os.path.join(MODEL_DIR, "confusion_matrix.png")):
            st.image(os.path.join(MODEL_DIR, "confusion_matrix.png"), caption="Confusion Matrix")
        if os.path.exists(os.path.join(MODEL_DIR, "error_distribution.png")):
            st.image(os.path.join(MODEL_DIR, "error_distribution.png"), caption="Error Distribution")

    with tabs[3]:
        st.header("Limitations")
        st.warning("""
        - **Not medical:** This app is for educational/research purposes only. It is not a medical device and must not be used for diagnosis or treatment.
        - **Unsupervised proxy:** Reconstruction error flags deviation from learned normal patterns, not specific arrhythmia types.
        - **Thresholding:** Simple percentile/mean+std thresholds have tradeoffs; test labels were not used for threshold selection.
        - **Dataset constraints:** Evaluated on preprocessed ECG5000 benchmark; may not generalize to raw clinical data.
        - **False results possible:** Overlap between normal and abnormal errors can cause false positives/negatives.
        """)


if __name__ == "__main__":
    main()
