import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

from .anomaly_detection import load_artifacts, get_errors
from .preprocess import prepare_dataloaders


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)


def map_labels(y):
    # Map to normal/non-normal: assume normal_label == 1.0 is normal
    y_norm = np.zeros_like(y)
    y_norm[np.isclose(y, 1.0)] = 0  # normal
    y_norm[~np.isclose(y, 1.0)] = 1  # anomaly/non-normal
    return y_norm


def evaluate(threshold: float = None):
    model, device, _ = load_artifacts()
    info = prepare_dataloaders()
    test_loader = info["test_loader"]
    y_test = info["y_test"]

    seg_err_test, pt_err_test = get_errors(test_loader, model, device)
    np.save(os.path.join(DATA_OUT, "test_seg_errors.npy"), seg_err_test)
    np.save(os.path.join(DATA_OUT, "test_pt_errors.npy"), pt_err_test)

    if threshold is None:
        try:
            with open(os.path.join(MODEL_DIR, "threshold.json")) as f:
                thresh_info = json.load(f)
            threshold = thresh_info["threshold"]
        except Exception:
            # compute from val
            from .anomaly_detection import select_threshold
            threshold = select_threshold()

    y_pred = (seg_err_test > threshold).astype(int)
    y_true = map_labels(y_test)

    # Metrics
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, seg_err_test)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)

    metrics = {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "auroc": float(auc),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot CM
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ["Normal (pred/true) mapped", "Anom"]
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Anom"])
    plt.yticks(tick_marks, ["Normal", "Anom"])
    thresh_cm = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh_cm else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()

    # Error distribution
    plt.figure(figsize=(8, 5))
    plt.hist(seg_err_test[y_true == 0], bins=50, alpha=0.6, label="Normal (true)")
    plt.hist(seg_err_test[y_true == 1], bins=50, alpha=0.6, label="Non-normal (true)")
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.xlabel("Segment MSE Reconstruction Error")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Reconstruction Error Distribution (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "error_distribution.png"))
    plt.close()

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
