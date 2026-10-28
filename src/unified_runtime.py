"""Small runtime helpers shared by training and evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .ptbxl_data import TARGET_CATEGORIES


def normal_mask_from_labels(labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    """Boolean mask selecting usable NORM-only records in a batch."""
    labels = labels.float()
    usable = label_mask.bool().any(dim=1)
    norm_only = (labels[:, 0] == 1.0) & (labels[:, 1:].sum(dim=1) == 0.0)
    return usable & norm_only


def masked_multilabel_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCEWithLogits over valid label entries only.

    Invalid/unknown entries are excluded via ``label_mask``; they are never
    treated as negative targets. If a batch has no valid entries the result is
    a zero tensor that still participates in autograd.
    """
    if logits.shape != targets.shape or logits.shape != label_mask.shape:
        raise ValueError("logits, targets and label_mask must have identical shapes")
    mask = label_mask.bool()
    if mask.sum() == 0:
        return logits.sum() * 0.0
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    elementwise = criterion(logits, targets.float())
    return elementwise[mask].mean()


def masked_normal_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    normal_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE over records explicitly marked as usable normal records.

    Abnormal records in a mixed batch contribute nothing, even if their
    reconstruction error is large.
    """
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target must have identical shapes")
    mask = normal_mask.bool().reshape(-1)
    if mask.numel() != reconstruction.shape[0]:
        raise ValueError("normal_mask must have one entry per record")
    if mask.sum() == 0:
        return reconstruction.sum() * 0.0
    squared = (reconstruction[mask] - target[mask]) ** 2
    return squared.mean()


def combined_objective(
    classification_loss: torch.Tensor,
    normal_reconstruction_loss: torch.Tensor,
    lambda_reconstruction: float,
) -> torch.Tensor:
    return classification_loss + float(lambda_reconstruction) * normal_reconstruction_loss


def class_pos_weight(labels: np.ndarray, label_mask: np.ndarray) -> np.ndarray:
    """Neg/pos ratio per category from training labels, clipped for stability."""
    labels = np.asarray(labels, dtype=float)
    mask = np.asarray(label_mask, dtype=bool)
    weights = np.ones(labels.shape[1], dtype=np.float32)
    for index in range(labels.shape[1]):
        valid = mask[:, index]
        positives = float(labels[valid, index].sum())
        negatives = float(valid.sum() - positives)
        if positives <= 0:
            weights[index] = 1.0
        else:
            weights[index] = float(np.clip(negatives / positives, 1.0, 20.0))
    return weights


@torch.no_grad()
def model_outputs_over_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, Any]:
    """Collect per-record losses, scores, labels and identifiers."""
    model.eval()
    errors: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    classification_losses: List[float] = []
    reconstruction_losses: List[float] = []
    ecg_ids: List[int] = []
    patient_ids: List[int] = []
    folds: List[int] = []
    for batch in loader:
        signal = batch["signal"].to(device)
        target_labels = batch["labels"].float().to(device)
        mask = batch["label_mask"].bool().to(device)
        output = model(signal)
        squared = (output.reconstruction - signal) ** 2
        per_record = squared.mean(dim=(1, 2))
        errors.append(per_record.cpu().numpy())
        logits.append(output.classification_logits.cpu().numpy())
        labels.append(target_labels.cpu().numpy())
        masks.append(mask.cpu().numpy())
        classification_losses.append(
            float(masked_multilabel_bce_loss(output.classification_logits, target_labels, mask).cpu())
        )
        normal_mask = normal_mask_from_labels(target_labels, mask)
        if normal_mask.any():
            reconstruction_losses.append(
                float(masked_normal_reconstruction_loss(output.reconstruction, signal, normal_mask).cpu())
            )
        ecg_ids.extend(int(v) for v in batch["ecg_id"].tolist())
        patient_ids.extend(int(v) for v in batch["patient_id"].tolist())
        folds.extend(int(v) for v in batch["fold"].tolist())
    scores = 1.0 / (1.0 + np.exp(-np.concatenate(logits).astype(np.float64))) if logits else np.empty((0, 5))
    return {
        "errors": np.concatenate(errors) if errors else np.empty((0,)),
        "logits": np.concatenate(logits) if logits else np.empty((0, 5)),
        "scores": scores,
        "labels": np.concatenate(labels) if labels else np.empty((0, 5)),
        "label_mask": np.concatenate(masks) if masks else np.empty((0, 5), dtype=bool),
        "classification_loss": (
            float(np.mean(classification_losses)) if classification_losses else None
        ),
        "normal_reconstruction_loss": (
            float(np.mean(reconstruction_losses)) if reconstruction_losses else None
        ),
        "ecg_ids": ecg_ids,
        "patient_ids": patient_ids,
        "folds": folds,
        "label_order": list(TARGET_CATEGORIES),
    }


def anomaly_reference_array(labels: np.ndarray, label_mask: np.ndarray) -> np.ndarray:
    from .ptbxl_data import anomaly_reference_from_labels

    return anomaly_reference_from_labels(np.asarray(labels), np.asarray(label_mask))
