"""Evaluation CLI for the unified PTB-XL model.

Thresholds are tuned during training on validation fold 9 and frozen into the
artifact. This module reports validation metrics and, when explicitly asked,
final test metrics. It refuses test evaluation if thresholds are not frozen.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .ptbxl_data import (
    TARGET_CATEGORIES,
    load_ptbxl_bundle,
    make_dataloader,
    select_frame,
    validate_patient_splits,
)
from .unified_artifacts import artifact_paths, load_unified_artifacts
from .unified_metrics import anomaly_metrics, classification_metrics
from .unified_runtime import anomaly_reference_array, model_outputs_over_loader
from .unified_visualize import plot_error_distribution

LOGGER = logging.getLogger("unified_evaluate")


def collect_split_outputs(
    loaded,
    bundle,
    split: str,
    *,
    data_dir: str,
    cache_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    max_records: Optional[int] = None,
    device=None,
) -> Dict[str, Any]:
    import torch

    if device is None:
        device = next(loaded.model.parameters()).device
    frame, labels, mask = select_frame(bundle, split)
    if max_records and max_records > 0:
        frame = frame.iloc[:max_records].reset_index(drop=True)
        labels = labels[:max_records]
        mask = mask[:max_records]
    loader = make_dataloader(
        frame,
        labels,
        mask,
        data_dir,
        loaded.preprocessing,
        loaded.normalization,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )
    outputs = model_outputs_over_loader(loaded.model, loader, device)
    outputs["frame"] = frame
    outputs["references"] = anomaly_reference_array(outputs["labels"], outputs["label_mask"])
    outputs["split"] = split
    return outputs


def evaluate_split(
    artifacts_path: str,
    data_dir: str,
    split: str,
    *,
    cache_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    max_records: Optional[int] = None,
    require_frozen: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    loaded = load_unified_artifacts(artifacts_path)
    frozen = bool(loaded.thresholds.get("frozen", False))
    if split == "test" and require_frozen and not frozen:
        raise RuntimeError(
            "Refusing final test evaluation because thresholds are not frozen. "
            "Run validation threshold tuning and freeze them first."
        )
    dataset_stats = loaded.metadata.get("dataset", {}).get("stats", {})
    min_likelihood = float(dataset_stats.get("min_likelihood", 0.0))
    bundle = load_ptbxl_bundle(data_dir, min_likelihood=min_likelihood)
    validate_patient_splits(
        bundle.metadata,
        bundle.metadata["split"].fillna("excluded").tolist(),
        raise_on_leakage=True,
    )
    outputs = collect_split_outputs(
        loaded,
        bundle,
        split,
        data_dir=data_dir,
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_records=max_records,
    )
    anomaly = anomaly_metrics(
        outputs["references"],
        outputs["errors"],
        float(loaded.thresholds["anomaly_threshold"]),
    )
    classification = classification_metrics(
        outputs["labels"],
        outputs["label_mask"],
        outputs["scores"],
        [float(v) for v in loaded.thresholds["class_thresholds"]],
    )
    result: Dict[str, Any] = {
        "split": split,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": loaded.paths.version_dir,
        "model_version": loaded.metadata.get("model_version"),
        "dataset": loaded.metadata.get("dataset", {}),
        "thresholds_frozen": frozen,
        "anomaly": anomaly,
        "classification": classification,
        "label_order": list(TARGET_CATEGORIES),
        "separation_note": (
            "Anomaly scores are reconstruction errors; condition scores are independent "
            "sigmoid outputs. They are on different scales and are not compared."
        ),
    }
    if save:
        os.makedirs(loaded.paths.version_dir, exist_ok=True)
        json_path = os.path.join(loaded.paths.version_dir, f"evaluation_{split}.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        plot_path = os.path.join(loaded.paths.version_dir, f"error_distribution_{split}.png")
        plot_error_distribution(
            outputs["errors"],
            float(loaded.thresholds["anomaly_threshold"]),
            outputs["references"],
            save_path=plot_path,
        )
        result["saved_json"] = json_path
        result["saved_plot"] = plot_path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the unified PTB-XL model")
    parser.add_argument("--artifacts", default=os.path.join("artifacts", "ptbxl_unified", "v1"))
    parser.add_argument("--data-dir", default=os.path.join("data", "ptbxl"))
    parser.add_argument("--split", choices=("val", "test", "both"), default="val")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None, help="Debug/smoke limit")
    parser.add_argument("--allow-unfrozen", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        result = evaluate_split(
            args.artifacts,
            args.data_dir,
            split,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_records=args.max_records,
            require_frozen=not args.allow_unfrozen,
            save=not args.no_save,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
