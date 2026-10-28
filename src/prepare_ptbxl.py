"""Validate a local PTB-XL download and optionally build a signal cache.

This tool never downloads data. Place the official PTB-XL 1.0.3 files under
``--data-dir`` first (see README), then run:

    python -m src.prepare_ptbxl --data-dir data/ptbxl --check-files 20
    python -m src.prepare_ptbxl --data-dir data/ptbxl --build-cache --cache-dir data/ptbxl_cache

Building the 100 Hz cache reads every WFDB record once and writes a
``(n_records, 12, 1000)`` float32 array (roughly 1 GB for the full dataset).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Sequence

from .ptbxl_data import (
    LEAD_ORDER,
    PTBXL_DATASET_VERSION,
    build_signal_cache,
    load_ptbxl_bundle,
    record_column_for_rate,
    sha256_file,
    validate_patient_splits,
)

LOGGER = logging.getLogger("prepare_ptbxl")


def check_waveform_files(bundle, data_dir: str, sampling_rate: int, limit: Optional[int]) -> Dict[str, Any]:
    column = record_column_for_rate(sampling_rate)
    frame = bundle.metadata if limit is None else bundle.metadata.iloc[:limit]
    missing = []
    for _index, row in frame.iterrows():
        path = row.get(column)
        if not isinstance(path, str) or not path:
            missing.append({"ecg_id": int(row["ecg_id"]), "reason": f"missing {column}"})
            continue
        root = os.path.join(data_dir, str(path).replace("\\", os.sep).replace("/", os.sep))
        if not (os.path.exists(root + ".hea") and os.path.exists(root + ".dat")):
            missing.append({"ecg_id": int(row["ecg_id"]), "path": root, "reason": "missing .hea/.dat"})
    return {
        "checked": int(len(frame)),
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "column": column,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PTB-XL and optionally build a signal cache")
    parser.add_argument("--data-dir", default=os.path.join("data", "ptbxl"))
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500), default=100)
    parser.add_argument("--min-likelihood", type=float, default=0.0)
    parser.add_argument("--check-files", type=int, default=10, help="Number of records whose .hea/.dat are checked (0=all)")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-records", type=int, default=None, help="Debug/smoke limit for cache building")
    parser.add_argument("--output", default=None, help="Optional JSON summary path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    data_dir = os.path.abspath(args.data_dir)
    LOGGER.info("Checking PTB-XL %s in %s", PTBXL_DATASET_VERSION, data_dir)
    bundle = load_ptbxl_bundle(data_dir, min_likelihood=args.min_likelihood)
    summary: Dict[str, Any] = {
        "dataset_version": PTBXL_DATASET_VERSION,
        "data_dir": data_dir,
        "sampling_rate": args.sampling_rate,
        "sequence_length": args.sampling_rate * 10,
        "lead_order": list(LEAD_ORDER),
        "stats": bundle.stats,
        "patient_split_check": validate_patient_splits(
            bundle.metadata,
            bundle.metadata["split"].fillna("excluded").tolist(),
            raise_on_leakage=True,
        ),
    }
    limit = None if args.check_files == 0 else args.check_files
    summary["waveform_files"] = check_waveform_files(bundle, data_dir, args.sampling_rate, limit)
    summary["source_sha256"] = {
        name: sha256_file(os.path.join(data_dir, name))
        for name in ("ptbxl_database.csv", "scp_statements.csv")
        if os.path.exists(os.path.join(data_dir, name))
    }
    if summary["waveform_files"]["missing_count"]:
        LOGGER.error(
            "%d checked record(s) are missing waveform files; training with WFDB reading will fail for them. "
            "Build a cache to skip and log invalid/missing records explicitly.",
            summary["waveform_files"]["missing_count"],
        )
    if args.build_cache:
        if not args.cache_dir:
            parser = build_parser()
            parser.error("--build-cache requires --cache-dir")
        summary["cache"] = build_signal_cache(
            bundle,
            data_dir,
            os.path.abspath(args.cache_dir),
            sampling_rate=args.sampling_rate,
            expected_length=args.sampling_rate * 10,
            max_records=args.max_records,
        )
        summary["cache"]["path"] = os.path.abspath(args.cache_dir)
    if args.output:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        LOGGER.info("Wrote summary to %s", output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
