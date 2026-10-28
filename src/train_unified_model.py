"""Staged, leakage-safe training for the unified PTB-XL model.

Stages
------
A. Reconstruction pretraining on training-partition NORM-only records.
B. Multilabel classifier training on records with usable diagnostic labels.
C. Optional joint fine-tuning: classification loss on all valid records plus
   reconstruction loss masked to NORM-only records in mixed batches.

Only folds 1-8 are used for fitting. Fold 9 is used for early stopping and
threshold tuning. Fold 10 is never touched by this module.

Examples
--------
    python -m src.train_unified_model all --data-dir data/ptbxl
    python -m src.train_unified_model stage-a --data-dir data/ptbxl
    python -m src.train_unified_model stage-b --data-dir data/ptbxl
    python -m src.train_unified_model stage-c --data-dir data/ptbxl
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .artifacts_helpers import (
    CURRENT_TRAINING_NOTES,
    build_stage_metadata,
    save_final_artifacts,
)
from .ptbxl_data import (
    PTBXLBundle,
    NormalizationStats,
    PreprocessingConfig,
    TARGET_CATEGORIES,
    artifact_dataset_info,
    fit_lead_normalization,
    load_ptbxl_bundle,
    load_signals_for_frame,
    make_dataloader,
    select_frame,
    validate_patient_splits,
)
from .unified_artifacts import (
    ARTIFACT_FORMAT_VERSION,
    artifact_paths,
    load_unified_artifacts,
    save_unified_artifacts,
    write_thresholds,
    default_threshold_payload,
)
from .unified_ecg_model import (
    MODEL_ARCHITECTURE,
    UnifiedECGModel,
    UnifiedModelConfig,
    count_parameters,
)
from .unified_metrics import (
    classification_metrics,
    select_anomaly_threshold,
    select_class_thresholds,
)
from .unified_runtime import (
    anomaly_reference_array,
    class_pos_weight,
    masked_multilabel_bce_loss,
    masked_normal_reconstruction_loss,
    combined_objective,
    model_outputs_over_loader,
    normal_mask_from_labels,
)

LOGGER = logging.getLogger("train_unified_model")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def preprocessing_from_args(args: argparse.Namespace) -> PreprocessingConfig:
    sequence_length = int(args.sampling_rate) * 10
    return PreprocessingConfig(
        sampling_rate=int(args.sampling_rate),
        sequence_length=sequence_length,
        normalization=str(args.normalization),
        clip_value=float(args.clip_value),
    )


def model_config_from_args(args: argparse.Namespace) -> UnifiedModelConfig:
    channels = tuple(int(part.strip()) for part in str(args.encoder_channels).split(",") if part.strip())
    return UnifiedModelConfig(
        sequence_length=int(args.sampling_rate) * 10,
        encoder_channels=channels,
        classifier_hidden=int(args.classifier_hidden),
        dropout=float(args.dropout),
    )


def _subset(frame, labels: np.ndarray, mask: np.ndarray, limit: Optional[int]):
    if limit is None or limit <= 0 or limit >= len(frame):
        return frame, labels, mask
    return (
        frame.iloc[:limit].reset_index(drop=True),
        labels[:limit],
        mask[:limit],
    )


def _loader(
    frame,
    labels: np.ndarray,
    mask: np.ndarray,
    bundle: PTBXLBundle,
    preprocessing: PreprocessingConfig,
    normalization: NormalizationStats,
    args: argparse.Namespace,
    *,
    shuffle: bool,
):
    return make_dataloader(
        frame,
        labels,
        mask,
        args.data_dir,
        preprocessing,
        normalization,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        cache_dir=args.cache_dir,
    )


def fit_normalization(
    bundle: PTBXLBundle,
    preprocessing: PreprocessingConfig,
    args: argparse.Namespace,
) -> NormalizationStats:
    if preprocessing.normalization != "train_lead_zscore":
        mode = preprocessing.normalization
        return NormalizationStats(
            mode=mode,
            mean=[0.0] * 12,
            std=[1.0] * 12,
            clip_value=preprocessing.clip_value,
        )
    frame, _labels, _mask = select_frame(bundle, "train", require_labels=True, normal_only=True)
    if len(frame) == 0:
        raise RuntimeError("No training-partition NORM-only records available to fit normalization")
    signals = load_signals_for_frame(
        frame,
        args.data_dir,
        preprocessing,
        cache_dir=args.cache_dir,
        max_records=args.norm_fit_records,
    )
    LOGGER.info("Fitting normalization on up to %s training-normal records", args.norm_fit_records or "all")
    stats = fit_lead_normalization(signals, clip_value=preprocessing.clip_value)
    return stats


def log_data_plan(args: argparse.Namespace) -> PTBXLBundle:
    bundle = load_ptbxl_bundle(args.data_dir, min_likelihood=args.min_likelihood)
    splits = bundle.metadata["split"].fillna("excluded").tolist()
    summary = validate_patient_splits(bundle.metadata, splits, raise_on_leakage=True)
    LOGGER.info("PTB-XL dataset version %s", bundle.dataset_version)
    LOGGER.info(
        "Records: %d total | usable labels %d | excluded %d",
        bundle.stats["n_records"],
        bundle.stats["n_usable_for_supervision"],
        bundle.stats["n_excluded_no_diagnostic_label"],
    )
    LOGGER.info("Patient-aware split summary: %s", json.dumps(summary))
    LOGGER.info("Positive counts: %s", bundle.stats["positive_counts"])
    for split in ("train", "val", "test"):
        frame = bundle.split_frame(split)
        normal = int((frame["anomaly_reference"] == 0).sum())
        LOGGER.info(
            "  %-5s records=%5d patients=%4d usable=%5d normal-only=%4d",
            split,
            len(frame),
            frame["patient_id"].nunique(),
            int(frame["has_target_labels"].sum()),
            normal,
        )
    return bundle


def _save_stage_a_checkpoint(
    path: str,
    model: UnifiedECGModel,
    preprocessing: PreprocessingConfig,
    normalization: NormalizationStats,
    bundle: PTBXLBundle,
    config: Dict[str, Any],
    history: Dict[str, Any],
) -> None:
    payload = {
        "checkpoint_type": "stage_a_reconstruction",
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "architecture": MODEL_ARCHITECTURE,
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "preprocessing": preprocessing.to_dict(),
        "normalization": normalization.to_dict(),
        "dataset": artifact_dataset_info(bundle, preprocessing),
        "training": {
            **config,
            "stage": "A",
            "objective": "masked reconstruction MSE on training-partition NORM-only records",
            "selection": "early stopping on validation-normal reconstruction loss (fold 9)",
        },
        "history": history,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_stage_a(
    path: str,
    device: torch.device,
) -> Tuple[UnifiedECGModel, PreprocessingConfig, NormalizationStats, Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Stage A checkpoint not found: {path}. Run stage-a first or pass --stage-a."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_type") != "stage_a_reconstruction":
        raise ValueError(f"{path} is not a stage-A reconstruction checkpoint")
    model = UnifiedECGModel(UnifiedModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    preprocessing = PreprocessingConfig.from_dict(checkpoint["preprocessing"])
    normalization = NormalizationStats.from_dict(checkpoint["normalization"])
    model.to(device)
    return model, preprocessing, normalization, checkpoint


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------


def run_stage_a(args: argparse.Namespace, bundle: Optional[PTBXLBundle] = None) -> str:
    set_seed(args.seed)
    device = resolve_device(args.device)
    bundle = bundle or log_data_plan(args)
    preprocessing = preprocessing_from_args(args)
    normalization = fit_normalization(bundle, preprocessing, args)
    LOGGER.info("Normalization mode=%s clip=%.2f", normalization.mode, normalization.clip_value)

    train_frame, train_labels, train_mask = select_frame(
        bundle, "train", require_labels=True, normal_only=True
    )
    val_frame, val_labels, val_mask = select_frame(
        bundle, "val", require_labels=True, normal_only=True
    )
    train_frame, train_labels, train_mask = _subset(
        train_frame, train_labels, train_mask, args.max_train_records
    )
    val_frame, val_labels, val_mask = _subset(
        val_frame, val_labels, val_mask, args.max_val_records
    )
    if len(train_frame) == 0:
        raise RuntimeError("No NORM-only records in the training partition; cannot run stage A")
    if len(val_frame) == 0:
        raise RuntimeError("No NORM-only records in the validation partition; cannot select threshold/early stop")

    train_loader = _loader(
        train_frame, train_labels, train_mask, bundle, preprocessing, normalization, args, shuffle=True
    )
    val_loader = _loader(
        val_frame, val_labels, val_mask, bundle, preprocessing, normalization, args, shuffle=False
    )
    config = model_config_from_args(args)
    model = UnifiedECGModel(config).to(device)
    LOGGER.info(
        "Stage A: %d train / %d val normal records | model params=%d | device=%s",
        len(train_frame), len(val_frame), count_parameters(model), device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 3)
    )

    checkpoint_path = os.path.join(args.out_dir, "stage_a.pt")
    best_val = float("inf")
    patience_left = int(args.patience)
    history: Dict[str, List[float]] = {"train_reconstruction_loss": [], "val_reconstruction_loss": []}
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            signal = batch["signal"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(signal)
            loss = torch.nn.functional.mse_loss(output.reconstruction, signal)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * signal.shape[0]
            seen += signal.shape[0]
        train_loss = running / max(seen, 1)

        model.eval()
        val_running = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                signal = batch["signal"].to(device)
                output = model(signal)
                loss = torch.nn.functional.mse_loss(output.reconstruction, signal)
                val_running += float(loss.cpu()) * signal.shape[0]
                val_seen += signal.shape[0]
        val_loss = val_running / max(val_seen, 1)
        scheduler.step(val_loss)
        history["train_reconstruction_loss"].append(train_loss)
        history["val_reconstruction_loss"].append(val_loss)
        LOGGER.info("Stage A epoch %d/%d train=%.6f val=%.6f", epoch, args.epochs, train_loss, val_loss)

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            patience_left = int(args.patience)
            _save_stage_a_checkpoint(
                checkpoint_path,
                model,
                preprocessing,
                normalization,
                bundle,
                {
                    "epochs_requested": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "best_epoch": epoch,
                    "best_val_reconstruction_loss": float(best_val),
                    "seed": int(args.seed),
                },
                history,
            )
            LOGGER.info("Saved stage A checkpoint: %s", checkpoint_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOGGER.info("Stage A early stopping at epoch %d", epoch)
                break
    LOGGER.info("Stage A complete. Best validation reconstruction loss: %.6f", best_val)
    return checkpoint_path


# ---------------------------------------------------------------------------
# Validation threshold tuning
# ---------------------------------------------------------------------------


def tune_thresholds_on_validation(
    version_dir: str,
    bundle: PTBXLBundle,
    preprocessing: PreprocessingConfig,
    normalization: NormalizationStats,
    model: UnifiedECGModel,
    args: argparse.Namespace,
    device: torch.device,
    *,
    freeze: bool = True,
) -> Dict[str, Any]:
    """Tune anomaly and class thresholds on fold 9 only, then optionally freeze."""
    val_frame, val_labels, val_mask = select_frame(bundle, "val", require_labels=True)
    val_frame, val_labels, val_mask = _subset(
        val_frame, val_labels, val_mask, args.max_val_records
    )
    if len(val_frame) == 0:
        raise RuntimeError("No validation records with usable labels for threshold tuning")
    loader = _loader(
        val_frame, val_labels, val_mask, bundle, preprocessing, normalization, args, shuffle=False
    )
    outputs = model_outputs_over_loader(model, loader, device)
    references = anomaly_reference_array(outputs["labels"], outputs["label_mask"])
    normal_errors = outputs["errors"][references == 0]
    abnormal_errors = outputs["errors"][references == 1]
    anomaly_threshold, anomaly_info = select_anomaly_threshold(
        normal_errors,
        abnormal_errors=abnormal_errors,
        method=args.anomaly_method,
        percentile=args.anomaly_percentile,
    )
    class_thresholds, class_info = select_class_thresholds(
        outputs["labels"], outputs["label_mask"], outputs["scores"]
    )
    payload = default_threshold_payload(
        anomaly_threshold,
        class_thresholds,
        anomaly_info=anomaly_info,
        class_info=class_info,
    )
    payload["frozen"] = bool(freeze)
    payload["frozen_utc"] = datetime.now(timezone.utc).isoformat() if freeze else None
    payload["validation_summary"] = {
        "n_records": int(len(val_frame)),
        "n_normal_reference": int((references == 0).sum()),
        "n_abnormal_reference": int((references == 1).sum()),
        "classification": classification_metrics(
            outputs["labels"], outputs["label_mask"], outputs["scores"], class_thresholds
        ),
    }
    write_thresholds(version_dir, payload)
    LOGGER.info(
        "Thresholds tuned on validation fold 9: anomaly=%.6f | classes=%s",
        anomaly_threshold, [round(v, 4) for v in class_thresholds],
    )
    return payload


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------


def _macro_f1_from_outputs(outputs: Dict[str, Any], thresholds: Sequence[float]) -> Optional[float]:
    metrics = classification_metrics(
        outputs["labels"], outputs["label_mask"], outputs["scores"], thresholds
    )
    return metrics["macro_f1"]


def run_stage_b(
    args: argparse.Namespace,
    bundle: Optional[PTBXLBundle] = None,
) -> str:
    set_seed(args.seed)
    device = resolve_device(args.device)
    bundle = bundle or log_data_plan(args)
    stage_a_path = args.stage_a or os.path.join(args.out_dir, "stage_a.pt")
    model, preprocessing, normalization, _checkpoint = _load_stage_a(stage_a_path, device)
    for parameter in model.decoder.parameters():
        parameter.requires_grad = False
    if args.freeze_encoder:
        model.freeze_encoder(True)
    LOGGER.info("Stage B: encoder frozen=%s, decoder frozen=True", bool(args.freeze_encoder))

    train_frame, train_labels, train_mask = select_frame(bundle, "train", require_labels=True)
    val_frame, val_labels, val_mask = select_frame(bundle, "val", require_labels=True)
    train_frame, train_labels, train_mask = _subset(
        train_frame, train_labels, train_mask, args.max_train_records
    )
    val_frame, val_labels, val_mask = _subset(
        val_frame, val_labels, val_mask, args.max_val_records
    )
    if len(train_frame) == 0 or len(val_frame) == 0:
        raise RuntimeError("Stage B requires usable labelled records in train and validation partitions")
    train_loader = _loader(
        train_frame, train_labels, train_mask, bundle, preprocessing, normalization, args, shuffle=True
    )
    val_loader = _loader(
        val_frame, val_labels, val_mask, bundle, preprocessing, normalization, args, shuffle=False
    )
    pos_weights = class_pos_weight(train_labels, train_mask)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    LOGGER.info("Training class pos_weight: %s", dict(zip(TARGET_CATEGORIES, pos_weights.round(3).tolist())))

    parameters: List[Dict[str, Any]] = [
        {"params": model.classifier.parameters(), "lr": float(args.lr)}
    ]
    if not args.freeze_encoder:
        parameters.append(
            {"params": model.encoder.parameters(), "lr": float(args.lr) * float(args.backbone_lr_scale)}
        )
    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay)

    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_left = int(args.patience)
    history: Dict[str, Any] = {"train_classification_loss": [], "val_classification_loss": [], "val_macro_f1": []}
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            signal = batch["signal"].to(device)
            target_labels = batch["labels"].float().to(device)
            mask = batch["label_mask"].bool().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(signal)
            loss = masked_multilabel_bce_loss(
                output.classification_logits, target_labels, mask, pos_weight=pos_weight_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
            running += float(loss.detach().cpu()) * signal.shape[0]
            seen += signal.shape[0]
        train_loss = running / max(seen, 1)
        outputs = model_outputs_over_loader(model, val_loader, device)
        macro_f1 = _macro_f1_from_outputs(outputs, [0.5] * len(TARGET_CATEGORIES))
        val_loss = outputs["classification_loss"]
        history["train_classification_loss"].append(train_loss)
        history["val_classification_loss"].append(val_loss)
        history["val_macro_f1"].append(macro_f1)
        score = macro_f1 if macro_f1 is not None else -float(val_loss or 0.0)
        LOGGER.info(
            "Stage B epoch %d/%d train_bce=%.6f val_bce=%s val_macro_f1=%s",
            epoch, args.epochs, train_loss,
            f"{val_loss:.6f}" if val_loss is not None else "n/a",
            f"{macro_f1:.4f}" if macro_f1 is not None else "n/a",
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_left = int(args.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOGGER.info("Stage B early stopping at epoch %d", epoch)
                break
    model.load_state_dict(best_state)
    model.eval()

    training_config = {
        "stage": "B",
        "encoder_initialized_from": stage_a_path,
        "encoder_frozen": bool(args.freeze_encoder),
        "decoder_frozen": True,
        "loss": "BCEWithLogitsLoss over valid label entries only (pos_weight from training partition)",
        "selection": "early stopping on validation macro-F1 (fold 9), default 0.5 thresholds",
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "backbone_lr_scale": float(args.backbone_lr_scale),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "label_masking": (
            "Records with no mapped diagnostic statement are excluded; for usable records, "
            "absence of a category is an annotated negative."
        ),
        "pos_weight": {category: float(pos_weights[i]) for i, category in enumerate(TARGET_CATEGORIES)},
        "notes": CURRENT_TRAINING_NOTES,
    }
    save_final_artifacts(
        args.out_dir,
        model,
        preprocessing=preprocessing,
        normalization=normalization,
        bundle=bundle,
        training_config=training_config,
        history=history,
        extra_metadata=build_stage_metadata("B", args),
    )
    if not args.skip_tune:
        tune_thresholds_on_validation(
            args.out_dir, bundle, preprocessing, normalization, model, args, device
        )
    LOGGER.info("Stage B complete. Model artifact: %s", artifact_paths(args.out_dir).model_path)
    return artifact_paths(args.out_dir).version_dir


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------


def run_stage_c(
    args: argparse.Namespace,
    bundle: Optional[PTBXLBundle] = None,
) -> str:
    set_seed(args.seed)
    device = resolve_device(args.device)
    bundle = bundle or log_data_plan(args)
    source = getattr(args, "stage_b", None) or args.out_dir
    loaded = load_unified_artifacts(source, device=device)
    model = loaded.model
    preprocessing = loaded.preprocessing
    normalization = loaded.normalization
    if args.freeze_encoder:
        model.freeze_encoder(True)
    if args.freeze_decoder:
        for parameter in model.decoder.parameters():
            parameter.requires_grad = False

    train_frame, train_labels, train_mask = select_frame(bundle, "train", require_labels=True)
    val_frame, val_labels, val_mask = select_frame(bundle, "val", require_labels=True)
    train_frame, train_labels, train_mask = _subset(
        train_frame, train_labels, train_mask, args.max_train_records
    )
    val_frame, val_labels, val_mask = _subset(
        val_frame, val_labels, val_mask, args.max_val_records
    )
    train_loader = _loader(
        train_frame, train_labels, train_mask, bundle, preprocessing, normalization, args, shuffle=True
    )
    val_loader = _loader(
        val_frame, val_labels, val_mask, bundle, preprocessing, normalization, args, shuffle=False
    )
    pos_weights = class_pos_weight(train_labels, train_mask)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    LOGGER.info(
        "Stage C: %d train / %d val labelled records | lambda_recon=%.3f | encoder frozen=%s",
        len(train_frame), len(val_frame), args.lambda_reconstruction, args.freeze_encoder,
    )

    parameter_groups = [
        {"params": model.classifier.parameters(), "lr": float(args.lr)},
    ]
    if not args.freeze_encoder:
        parameter_groups.append(
            {"params": model.encoder.parameters(), "lr": float(args.lr) * float(args.backbone_lr_scale)}
        )
    if not args.freeze_decoder:
        parameter_groups.append(
            {"params": model.decoder.parameters(), "lr": float(args.lr) * float(args.backbone_lr_scale)}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_left = int(args.patience)
    history: Dict[str, Any] = {
        "train_total_loss": [],
        "train_classification_loss": [],
        "train_normal_reconstruction_loss": [],
        "val_classification_loss": [],
        "val_normal_reconstruction_loss": [],
        "val_total_loss": [],
        "val_macro_f1": [],
    }
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        totals = {"total": 0.0, "classification": 0.0, "reconstruction": 0.0, "recon_batches": 0}
        seen = 0
        for batch in train_loader:
            signal = batch["signal"].to(device)
            target_labels = batch["labels"].float().to(device)
            mask = batch["label_mask"].bool().to(device)
            normal_mask = normal_mask_from_labels(target_labels, mask)
            optimizer.zero_grad(set_to_none=True)
            output = model(signal)
            cls_loss = masked_multilabel_bce_loss(
                output.classification_logits, target_labels, mask, pos_weight=pos_weight_tensor
            )
            recon_loss = masked_normal_reconstruction_loss(
                output.reconstruction, signal, normal_mask
            )
            total = combined_objective(cls_loss, recon_loss, args.lambda_reconstruction)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
            batch_size = signal.shape[0]
            totals["total"] += float(total.detach().cpu()) * batch_size
            totals["classification"] += float(cls_loss.detach().cpu()) * batch_size
            if bool(normal_mask.any()):
                totals["reconstruction"] += float(recon_loss.detach().cpu()) * int(normal_mask.sum())
                totals["recon_batches"] += 1
            seen += batch_size
        outputs = model_outputs_over_loader(model, val_loader, device)
        macro_f1 = _macro_f1_from_outputs(outputs, [0.5] * len(TARGET_CATEGORIES))
        val_cls = outputs["classification_loss"]
        val_recon = outputs["normal_reconstruction_loss"]
        val_total = (
            (val_cls or 0.0)
            + float(args.lambda_reconstruction) * (val_recon or 0.0)
        )
        history["train_total_loss"].append(totals["total"] / max(seen, 1))
        history["train_classification_loss"].append(totals["classification"] / max(seen, 1))
        history["train_normal_reconstruction_loss"].append(
            totals["reconstruction"] / max(1, seen) if totals["recon_batches"] else None
        )
        history["val_classification_loss"].append(val_cls)
        history["val_normal_reconstruction_loss"].append(val_recon)
        history["val_total_loss"].append(val_total)
        history["val_macro_f1"].append(macro_f1)
        score = macro_f1 if macro_f1 is not None else -val_total
        LOGGER.info(
            "Stage C epoch %d/%d total=%.6f cls=%.6f recon=%s | val_f1=%s val_total=%.6f",
            epoch, args.epochs, history["train_total_loss"][-1],
            history["train_classification_loss"][-1],
            f"{history['train_normal_reconstruction_loss'][-1]:.6f}"
            if history["train_normal_reconstruction_loss"][-1] is not None else "n/a",
            f"{macro_f1:.4f}" if macro_f1 is not None else "n/a",
            val_total,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_left = int(args.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOGGER.info("Stage C early stopping at epoch %d", epoch)
                break
    model.load_state_dict(best_state)
    model.eval()

    training_config = {
        "stage": "C",
        "initialized_from": source,
        "encoder_frozen": bool(args.freeze_encoder),
        "decoder_frozen": bool(args.freeze_decoder),
        "loss": "classification_loss + lambda_reconstruction * normal_reconstruction_loss",
        "classification_loss": "BCEWithLogitsLoss on valid label entries only",
        "reconstruction_loss": "MSE on NORM-only records in each mixed batch; abnormal records masked out",
        "lambda_reconstruction": float(args.lambda_reconstruction),
        "selection": "early stopping on validation macro-F1 (fold 9)",
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "backbone_lr_scale": float(args.backbone_lr_scale),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "pos_weight": {category: float(pos_weights[i]) for i, category in enumerate(TARGET_CATEGORIES)},
        "trade_offs": (
            "Sharing the encoder lets reconstruction and classification influence each other. "
            "Normal-only reconstruction may bias features toward NORM morphology, while "
            "classification gradients may reduce reconstruction specificity. Compare stage B "
            "and stage C on validation data; use the same held-out test partition only once."
        ),
        "notes": CURRENT_TRAINING_NOTES,
    }
    save_final_artifacts(
        args.out_dir,
        model,
        preprocessing=preprocessing,
        normalization=normalization,
        bundle=bundle,
        training_config=training_config,
        history=history,
        extra_metadata=build_stage_metadata("C", args),
    )
    if not args.skip_tune:
        tune_thresholds_on_validation(
            args.out_dir, bundle, preprocessing, normalization, model, args, device
        )
    LOGGER.info("Stage C complete. Model artifact: %s", artifact_paths(args.out_dir).model_path)
    return artifact_paths(args.out_dir).version_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=os.path.join("data", "ptbxl"), help="PTB-XL 1.0.3 directory")
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "ptbxl_unified", "v1"))
    parser.add_argument("--cache-dir", default=None, help="Optional prebuilt signal cache")
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500), default=100)
    parser.add_argument("--normalization", choices=("train_lead_zscore", "per_record_zscore", "none"), default="train_lead_zscore")
    parser.add_argument("--clip-value", type=float, default=5.0)
    parser.add_argument("--min-likelihood", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-channels", default="32,64,128,256")
    parser.add_argument("--classifier-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-train-records", type=int, default=None, help="Debug/smoke limit")
    parser.add_argument("--max-val-records", type=int, default=None, help="Debug/smoke limit")
    parser.add_argument("--norm-fit-records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--anomaly-method", choices=("percentile", "mean_std", "f1"), default="percentile")
    parser.add_argument("--anomaly-percentile", type=float, default=95.0)
    parser.add_argument("--skip-tune", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the unified PTB-XL shared-encoder model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_a = subparsers.add_parser("stage-a", help="Reconstruction pretraining on NORM-only records")
    add_common_arguments(stage_a)

    stage_b = subparsers.add_parser("stage-b", help="Supervised multilabel classifier training")
    add_common_arguments(stage_b)
    stage_b.add_argument("--stage-a", default=None)
    stage_b.add_argument("--freeze-encoder", action="store_true")

    stage_c = subparsers.add_parser("stage-c", help="Joint classification + normal reconstruction fine-tuning")
    add_common_arguments(stage_c)
    stage_c.add_argument("--stage-b", default=None)
    stage_c.add_argument("--freeze-encoder", action="store_true")
    stage_c.add_argument("--freeze-decoder", action="store_true")
    stage_c.add_argument("--lambda-reconstruction", type=float, default=0.5)

    all_parser = subparsers.add_parser("all", help="Run stages A, B and C")
    add_common_arguments(all_parser)
    all_parser.add_argument("--stage-a", default=None)
    all_parser.add_argument("--freeze-encoder-b", action="store_true")
    all_parser.add_argument("--freeze-encoder-c", action="store_true")
    all_parser.add_argument("--freeze-decoder", action="store_true")
    all_parser.add_argument("--lambda-reconstruction", type=float, default=0.5)

    tune = subparsers.add_parser("tune", help="Recompute validation-only thresholds for an existing artifact")
    add_common_arguments(tune)

    return parser


def _run_tune(args: argparse.Namespace) -> str:
    device = resolve_device(args.device)
    bundle = log_data_plan(args)
    loaded = load_unified_artifacts(args.out_dir, device=device)
    tune_thresholds_on_validation(
        args.out_dir,
        bundle,
        loaded.preprocessing,
        loaded.normalization,
        loaded.model,
        args,
        device,
    )
    return loaded.paths.version_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "out_dir"):
        args.out_dir = os.path.abspath(args.out_dir)
    if getattr(args, "cache_dir", None):
        args.cache_dir = os.path.abspath(args.cache_dir)
    if hasattr(args, "data_dir"):
        args.data_dir = os.path.abspath(args.data_dir)
    LOGGER.info("Device requested: %s | output: %s", args.device, args.out_dir)
    bundle = log_data_plan(args)
    if args.command == "stage-a":
        run_stage_a(args, bundle)
    elif args.command == "stage-b":
        run_stage_b(args, bundle)
    elif args.command == "stage-c":
        run_stage_c(args, bundle)
    elif args.command == "tune":
        _run_tune(args)
    elif args.command == "all":
        run_stage_a(args, bundle)
        args.freeze_encoder = bool(args.freeze_encoder_b)
        run_stage_b(args, bundle)
        args.freeze_encoder = bool(args.freeze_encoder_c)
        run_stage_c(args, bundle)
    else:  # pragma: no cover
        parser.error(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
