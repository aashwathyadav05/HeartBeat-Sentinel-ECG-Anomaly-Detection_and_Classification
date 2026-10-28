# HeartBeat Sentinel: Explainable ECG Anomaly Detection

HeartBeat Sentinel is an **educational research prototype** for explainable ECG
analysis. It contains two deliberately separate workflows:

| Workflow | Signal | Model | Purpose |
| --- | --- | --- | --- |
| **Unified PTB-XL** | 12 leads × 1000 samples (10 s at 100 Hz) | Shared 1D-CNN encoder + reconstruction decoder + 5-output classifier | Reconstruction-based anomaly score **and** multilabel condition scores |
| **Legacy ECG5000** | 1 lead × 140 samples | 1D convolutional autoencoder | Reconstruction-based anomaly score on the original benchmark |

> **Not a medical device.** Outputs are research/education signals, not diagnoses.
> An anomaly score measures deviation from learned normal reconstruction; a
> condition score is not a calibrated clinical probability.

## Unified PTB-XL model

### Shared architecture

A single strided 1D convolutional encoder feeds two task heads:

```
(batch, 12, 1000)
        │
   shared encoder      4 strided Conv1d stages: 32 → 64 → 128 → 256 channels
        │              followed by concatenated mean/max temporal pooling
        ├── decoder    ConvTranspose1d + interpolation → (batch, 12, 1000)
        └── classifier 2-layer MLP → 5 logits
```

The classifier produces **five independent sigmoid scores** in the fixed order
`[NORM, MI, STTC, CD, HYP]`. It is multilabel by design: several categories can
coexist, and a softmax is never applied.

### Anomaly score vs. classification

These outputs answer different questions and must not be conflated:

* **Anomaly score** — mean squared reconstruction error in normalized units,
  compared with one frozen threshold. It says "this ECG reconstructs less like
  the learned normal distribution", not *which* condition is present.
* **Condition scores** — five independent sigmoid values compared with five
  frozen per-category thresholds. They estimate the presence of PTB-XL
  diagnostic superclasses, but are not calibrated probabilities.

For anomaly reference labels, a record is *normal* only when `NORM` is positive
and none of `MI/STTC/CD/HYP` is positive. Any of those four makes it *abnormal*.
Records without a usable five-class label set have no anomaly reference.

### Training stages

1. **Stage A — normal-only reconstruction pretraining.** Fits the shared encoder
   and decoder with MSE on training-fold records that are NORM-only.
2. **Stage B — supervised multilabel training.** Trains the classifier with
   `BCEWithLogitsLoss` over valid label entries only. The decoder stays frozen;
   the encoder can be frozen or fine-tuned at a smaller learning rate.
3. **Stage C — optional joint fine-tuning.**
   `classification_loss + lambda_reconstruction * normal_reconstruction_loss`,
   where the reconstruction term uses an explicit normal-only mask.

Unknown or unusable labels are **never silently treated as negatives**. A record
is usable only if at least one diagnostic SCP statement maps to one of the five
targets and passes `--min-likelihood`. Records with no diagnostic statement,
only non-diagnostic statements, or only out-of-scope diagnostic classes are
excluded and logged.

### Dataset, folds and preprocessing

* Dataset: **PTB-XL 1.0.3**, physical ECG signals in mV.
* Default rate: **100 Hz** (`records100/`), exactly **1000 samples** per lead.
* Leads: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
* Splits use the official `strat_fold` column:
  * folds **1–8** → train
  * fold **9** → validation
  * fold **10** → test (untouched until final evaluation)
* Patient leakage is checked explicitly across partitions.
* Default normalization is **per-lead z-score fitted only on training-fold
  NORM-only records**, clipped at ±5, and saved in the artifact for reuse.
* Thresholds are tuned on validation fold 9 only and frozen before test
  evaluation. The evaluator refuses unfrozen test runs.

PTB-XL is **not downloaded and training is not started automatically**. Place
the official files yourself:

```text
data/ptbxl/
├── ptbxl_database.csv
├── scp_statements.csv
├── records100/...
└── records500/...        # optional when using 100 Hz
```

Download PTB-XL 1.0.3 from PhysioNet
(<https://physionet.org/content/ptb-xl/1.0.3/>), extract the archive, and point
`--data-dir` at the directory containing `ptbxl_database.csv`.

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd.exe: .venv\Scripts\activate.bat
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## PTB-XL workflow

### 1. Validate the local dataset and optionally cache signals

```bash
python -m src.prepare_ptbxl --data-dir data/ptbxl --check-files 20
```

Optionally build a reusable 100 Hz cache (roughly 1 GB for the full dataset):

```bash
python -m src.prepare_ptbxl --data-dir data/ptbxl --build-cache --cache-dir data/ptbxl_cache
```

These commands never download data. `--cache-dir` can also be passed to training
and evaluation.

### 2. Train the unified model

Run all three stages:

```bash
python -m src.train_unified_model all --data-dir data/ptbxl --out-dir artifacts/ptbxl_unified/v1
```

Or run stages explicitly:

```bash
python -m src.train_unified_model stage-a --data-dir data/ptbxl --out-dir artifacts/ptbxl_unified/v1

python -m src.train_unified_model stage-b --data-dir data/ptbxl --out-dir artifacts/ptbxl_unified/v1 --stage-a artifacts/ptbxl_unified/v1/stage_a.pt

python -m src.train_unified_model stage-c --data-dir data/ptbxl --out-dir artifacts/ptbxl_unified/v1 --stage-b artifacts/ptbxl_unified/v1 --lambda-reconstruction 0.5
```

Useful flags include `--epochs`, `--patience`, `--batch-size`, `--lr`,
`--encoder-channels`, `--normalization`, `--min-likelihood`, `--device`,
`--max-train-records`/`--max-val-records` (smoke tests), and
`--anomaly-percentile`. Stage B/C add `--freeze-encoder`; Stage C also accepts
`--freeze-decoder`.

### 3. Re-tune thresholds on validation only

```bash
python -m src.train_unified_model tune --data-dir data/ptbxl --out-dir artifacts/ptbxl_unified/v1
```

### 4. Evaluate validation and test

```bash
python -m src.unified_evaluate --artifacts artifacts/ptbxl_unified/v1 --data-dir data/ptbxl --split both
```

Evaluation writes `evaluation_val.json` / `evaluation_test.json` plus plots into
the artifact version directory. Test evaluation requires frozen thresholds; use
`--allow-unfrozen` only for debugging, never for reported test metrics.

### 5. Artifacts

Each version directory contains everything needed for reproducible inference:

```text
artifacts/ptbxl_unified/
├── latest.json                 # pointer to the active version
└── v1/
    ├── model.pt                # state dict + model config + format version
    ├── metadata.json           # label order, preprocessing, normalization, dataset,
    │                           # dataset version and training config
    ├── thresholds.json         # frozen anomaly + five class thresholds
    ├── training_history.json
    └── evaluation_*.json       # optional
```

`artifacts/`, datasets, checkpoints and caches are gitignored and must not be
committed.

## PTB-XL classification notebook

[`notebooks/PTB_XL_Multilabel_Classification.ipynb`](notebooks/PTB_XL_Multilabel_Classification.ipynb)
uses the production label mapper, WFDB reader, preprocessing, model and
inference service. It includes metadata/label EDA, fold and patient checks,
real waveform visualization, an opt-in real-record gradient smoke cell, and
artifact-gated inference. It does not start full training. Run it from the
repository root with the project environment active; training curves and test
metrics remain pending until the CLI workflow has actually produced them.

## Applications

Both Flask and Streamlit call the same inference services
(`src.unified_inference` and `src.ecg5000_inference`); neither reimplements
prediction logic.

### Flask

```bash
python -m web.app
```

Open <http://127.0.0.1:5000>. The homepage immediately shows the five-condition
guide and the unified analyzer. Relevant endpoints:

* `GET /api/health` — legacy ECG5000 status
* `GET /api/unified/health`, `/api/unified/conditions`, `/api/unified/metrics`
* `GET /api/unified/examples`, `/api/unified/example/<ecg_id>`
* `POST /api/unified/predict` — JSON signal
* `POST /api/unified/predict-file` — uploaded `.npy`, `.csv`, `.txt` or `.json`
* `POST /api/unified/explain` — occlusion sensitivity
* Legacy `/api/predict`, `/api/metrics`, `/api/examples` keep their formats

### Streamlit

```bash
streamlit run app.py
```

### Input contract

The unified model accepts a finite array shaped `(12, 1000)` or
`(1, 12, 1000)` at 100 Hz. A 140-sample ECG5000 input is rejected with a clear
error and must use the legacy workflow instead. Full-rate (500 Hz) input is not
silently resampled.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `HEARTBEAT_UNIFIED_ARTIFACTS` | Artifact root/version used by inference |
| `HEARTBEAT_PTBXL_DIR` | Local PTB-XL directory for examples |
| `HEARTBEAT_PTBXL_CACHE` | Prebuilt signal cache directory |

If no unified artifact exists, the UI and unified APIs report a clear
"unified model unavailable" status; they do not crash or auto-train.

## Legacy ECG5000 workflow

The original 140-sample pipeline remains operational with its existing weights
(`models/cae_best.pt`), threshold (`models/threshold.json`) and metrics
(`models/metrics.json`).

* Download ECG5000 from
  [UCR/TimeseriesClassification](https://timeseriesclassification.com/description.php?Dataset=ECG5000)
  and place `ECG5000_TRAIN.txt` / `ECG5000_TEST.txt` in `dataset/`.
* Preprocess/train/evaluate:

```bash
python -m src.train
python -m src.evaluate
```

* Legacy inference is exposed through `/api/predict` and the Streamlit/Flask
  "ECG5000" tabs.

## Tests

Most tests use synthetic metadata, signals and tiny artifacts. One real-record
smoke test reads `records100/00000/00001_lr` when the local dataset is present.
The suite performs no full model training:

```bash
python -m pytest -q
```

## Project structure

```text
src/
├── ptbxl_data.py            # metadata, labels, folds, normalization, dataset
├── prepare_ptbxl.py         # local validation + optional signal cache
├── unified_ecg_model.py     # shared encoder, decoder, multilabel classifier
├── unified_artifacts.py     # versioned save/load and compatibility checks
├── unified_inference.py     # shared unified inference service
├── unified_runtime.py       # masked losses and batched outputs
├── unified_metrics.py       # anomaly/classification metrics and tuning
├── unified_explain.py       # occlusion sensitivity
├── unified_visualize.py     # plots
├── train_unified_model.py   # stages A/B/C + tune CLI
├── unified_evaluate.py      # validation/test evaluation CLI
├── ecg5000_inference.py     # legacy shared inference service
├── data_loader.py, model.py, preprocess.py, train.py, anomaly_detection.py,
│   evaluate.py, visualize.py  # legacy ECG5000 pipeline
web/                         # Flask app and static frontend
app.py                       # Streamlit dashboard
tests/                       # synthetic unit/integration tests
```

## Limitations

See [MODEL_CARD.md](MODEL_CARD.md) for the current data, label-policy, intended-use,
and evaluation status. In particular, the closed-world treatment of absent
target categories has not yet been validated against real PTB-XL records in this
checkout.

* Educational research prototype; not validated or approved for clinical use.
* PTB-XL data is local and gitignored rather than included in the repository;
  no trained unified artifact is currently included.
* Condition scores are multilabel model outputs, not calibrated probabilities and
  not a diagnosis.
* Anomaly detection only measures reconstruction deviation; it cannot localize or
  name a condition by itself.
* Performance depends on PTB-XL label quality, fold composition, preprocessing and
  threshold calibration. External cohorts are untested.
* The legacy ECG5000 model is a separate 140-sample benchmark and must not be
  applied to 12-lead PTB-XL recordings.

## Citations

* Wagner et al., **PTB-XL, a large publicly available electrocardiography dataset**,
  Scientific Data (2020). <https://physionet.org/content/ptb-xl/1.0.3/>
* Goldberger et al., **PhysioBank, PhysioToolkit, and PhysioNet** (2000).
* Dau et al., **The UCR Time Series Archive**.
  <https://timeseriesclassification.com/description.php?Dataset=ECG5000>
