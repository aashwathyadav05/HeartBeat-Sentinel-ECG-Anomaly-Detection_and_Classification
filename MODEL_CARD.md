# HeartBeat Sentinel model card

## Status and intended use

This repository contains an educational research implementation for ECG
reconstruction anomaly scoring and multilabel prediction of PTB-XL diagnostic
superclasses (NORM, MI, STTC, CD, HYP). It is not a medical device and must not
be used for diagnosis, triage, or treatment decisions.

The unified architecture has been trained locally on PTB-XL 1.0.3. The
checkpoint and evaluation outputs are local, gitignored artifacts. A real
held-out test evaluation is recorded below. These results establish a working
research pipeline, not clinical validity.

## Data and labels

The intended source is PTB-XL 1.0.3. The configured split is folds 1–8 for
training, fold 9 for validation, and fold 10 for final testing. The code checks
patient separation, fits signal normalization on training-fold normal records,
and selects thresholds using validation data before test evaluation.

The current target builder uses a closed-world convention: a record is usable
when at least one diagnostic SCP statement maps to a target superclass; absent
target superclasses are then encoded as negatives. Records without a mapped
diagnostic target are masked/excluded. This convention is documented in the
implementation and the metadata mapping has been executed on the local PTB-XL
1.0.3 copy. The semantic validity of treating every absent superclass as
negative has not been independently validated against the underlying clinical
annotations. Treat label completeness and ambiguity as unresolved scientific
limitations.

## Inputs and outputs

The default unified input is a finite 12 × 1000 float array at 100 Hz in the
fixed lead order `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`. Outputs
include a reconstruction-based deviation score and five independent
classification scores. The anomaly score does not name a disease, and the
classification scores are not established as calibrated clinical probabilities.

## Explainability and risks

The implementation provides reconstruction comparisons and temporal occlusion
attributions. These describe model sensitivity to input regions; they do not
establish causal or clinically meaningful findings. Dataset imbalance, label
noise, acquisition differences, preprocessing, threshold selection, and domain
shift may affect results. External-cohort performance is unknown.

## Evaluation status

The full automated suite passed with the real-record smoke test enabled. The
smoke test read `records100/00000/00001_lr` and verified a finite `(12, 1000)`
signal at 100 Hz. Preparation found 21,799 metadata records, 21,388 usable
mapped-label records, and no patient overlap across folds; 20 sampled waveform
pairs were present, and a later full path check found all 21,799 `.hea`/`.dat`
pairs. A two-record real-data training-path smoke check verified forward
shapes, finite losses/gradients, and an optimizer step.

Training used the CPU environment with seed 42 and the official folds. Stage A
early-stopped at epoch 17; Stage B early-stopped at epoch 22; Stage C
early-stopped at epoch 11. Stage C selected thresholds on validation fold 9.
The final checkpoint is Stage C (`artifacts/ptbxl_unified/v1/model.pt`). The
test fold was evaluated once afterward with frozen thresholds; 2,158 records
were evaluated and 40 records without usable labels were excluded.

| Held-out classification metric | Result |
| --- | ---: |
| Macro-F1 | 0.7073 |
| Micro-F1 | 0.7491 |

| Category | Support | F1 | AUROC | Validation threshold |
| --- | ---: | ---: | ---: | ---: |
| NORM | 963 | 0.8452 | 0.9285 | 0.4783 |
| MI | 550 | 0.6920 | 0.8890 | 0.6591 |
| STTC | 521 | 0.7479 | 0.9289 | 0.4891 |
| CD | 496 | 0.7028 | 0.8859 | 0.4248 |
| HYP | 262 | 0.5486 | 0.8553 | 0.8160 |

The anomaly proxy had AUROC 0.6355, precision 0.8450, recall 0.2319, and
F1 0.3640 at the validation-selected threshold 0.5503. The low recall means
this operating point missed many records labeled abnormal by the project's
proxy definition. The proxy defines abnormal as any MI/STTC/CD/HYP label and
normal as NORM only; it is not a comprehensive clinical abnormality label.

Stage B reached validation macro-F1 0.7098 at epoch 14; Stage C's best logged
validation macro-F1 was 0.7016 at epoch 3. The final test score must not be
used to choose between stages or retune thresholds. No independent baseline
comparison or external-cohort evaluation has been run. These dataset-specific
metrics are not evidence of clinical safety or generalization.
