# HeartBeat Sentinel: Explainable ECG Anomaly Detection Using a 1D Convolutional Autoencoder

## Project Overview

HeartBeat Sentinel is an educational research prototype for explainable ECG anomaly detection. A 1D Convolutional Autoencoder (CAE) is trained exclusively on normal ECG segments to learn a compact representation of "normal" cardiac rhythm. At inference time, segments that deviate from learned normal patterns produce higher reconstruction error and are flagged as potential anomalies. The system provides reconstruction visualizations, point-wise error heatmaps, and an interactive Flask+Streamlit dashboard.

**Important:** This is an educational prototype, not a clinically validated diagnostic device. Do not use for medical diagnosis or treatment decisions.

## Dataset (ECG5000)

Source: [UCR Time Series Classification - ECG5000](https://timeseriesclassification.com/description.php?Dataset=ECG5000)

**Observed structure (from local files):**
- Format: Space-delimited text files (`.txt`). First column = class label (as float string), remaining columns = waveform samples.
- Signal length: **140 samples** per ECG segment (141 columns total).
- Training set: **500** segments (classes in train: `1.0000000e+00` appears **292** times, `2.0000000e+00` **177**, `4.0000000e+00` **19**, `3.0000000e+00` **10**, `5.0000000e+00` **2`).
- Test set: **4500** segments (class distribution as observed: `1.0` **2627**, `2.0` **1590**, `4.0` **175**, `3.0` **86**, `5.0` **22`).
- The majority class in this dataset is typically considered the "normal" class (class `1` in the original labeling as distributed here). Treat class labels as dataset-provided ground truth for evaluation; the autoencoder is trained only on normal-labelled segments.

**Preprocessing already present:** The ECG5000 is a preprocessed, fixed-length time series benchmark from the UCR archive (segments extracted/normalized as provided). Our pipeline verifies dimensions, handles loading, splits normal training data into train/val, and never uses test labels during training/threshold selection.

### Download Instructions
If the dataset files are missing, obtain ECG5000 from the UCR/TimeseriesClassification page. Place the following files into `dataset/`:
- `ECG5000_TRAIN.txt`
- `ECG5000_TEST.txt`
- (Optionally also keep `.arff`, `.ts` if present, but `.txt` is used by default.)

## Project Structure
```
HeartBeat-Sentinel/
├── data/                 # processed artifacts (not committed by default)
├── notebooks/            # exploratory notebooks
├── models/               # saved checkpoints (not committed by default)
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── anomaly_detection.py
│   ├── evaluate.py
│   └── visualize.py
├── app.py                # Streamlit dashboard
├── requirements.txt
├── web/
│   ├── inference.py
│   ├── templates/        # index.html
│   ├── static/           # HTML,CSS pages
│   ├── app.py            # Flask application
├── README.md
└── .gitignore
```

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1) Preprocess and inspect
```bash
python -c "
from src.preprocess import prepare_dataloaders, visualize_samples
dl = prepare_dataloaders()
visualize_samples(n=3)
"
```

### 2) Train the autoencoder (on normal segments only)
```bash
python src/train.py
```
This will save the best checkpoint to `models/cae_best.pt` and training history to `models/history.json`. Plots are saved under `models/` (e.g. `loss_curves.png`).

### 3) Anomaly scoring and threshold selection (validation normal)
```bash
python -c "
from src.anomaly_detection import select_threshold, load_artifacts
art = load_artifacts()
thresh = select_threshold(method='percentile', percentile=95)
print('threshold:', thresh)
"
```

### 4) Evaluate on untouched test set
```bash
python src/evaluate.py
```
Saves metrics to `models/metrics.json` and confusion matrix plot to `models/confusion_matrix.png`.

### 5) Launch the dashboard
```bash
streamlit run app.py
```

## Methodology (Brief)

- **Unsupervised learning:** Autoencoder learns to reconstruct normal ECGs; anomalies yield larger reconstruction error.
- **1D CAE:** Encoder downsamples via Conv1d+ReLU, decoder upsamples via ConvTranspose1d/upsample+Conv1d to match input length exactly.
- **Loss:** MSE `(1/N)∑(x_i - x̂_i)^2`. Segment-level score is mean squared reconstruction error.
- **Threshold:** Chosen on normal validation samples (e.g. 95th percentile) — not optimized on test labels.
- **Explainability:** Original vs reconstructed vs point-wise squared error with segment score and threshold.

## Limitations
- Educational prototype only; not for clinical use.
- Thresholding is simple and may be sensitive to noise/class overlap.
- ECG5000 is a preprocessed benchmark; performance may not generalize to raw clinical ECGs.
- Anomaly score indicates deviation from learned normal patterns, not a specific diagnosis.

## Citations
- Dau et al., The UCR Time Series Archive. [timeseriesclassification.com](https://timeseriesclassification.com/)
- ECG5000 dataset description as hosted by UCR/TimeseriesClassification.
