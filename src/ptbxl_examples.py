"""Read-only PTB-XL examples shared by Flask and Streamlit.

No training or downloading happens here. Dataset labels returned alongside a
signal are for evaluation context only and are never presented as predictions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ptbxl_data import (
    TARGET_CATEGORIES,
    PTBXLDataError,
    PreprocessingConfig,
    load_ptbxl_bundle,
    read_ptbxl_signal,
    read_signal_cache,
)


def list_test_examples(data_dir: str, limit: int = 20) -> List[Dict[str, Any]]:
    bundle = load_ptbxl_bundle(data_dir)
    frame = bundle.metadata[
        (bundle.metadata["split"] == "test") & (bundle.metadata["has_target_labels"])
    ]
    examples: List[Dict[str, Any]] = []
    for position, row in frame.head(limit).iterrows():
        labels = bundle.labels[int(position)]
        mask = bundle.label_mask[int(position)]
        examples.append(
            {
                "ecg_id": int(row["ecg_id"]),
                "true_categories": [
                    category
                    for index, category in enumerate(TARGET_CATEGORIES)
                    if bool(mask[index]) and float(labels[index]) == 1.0
                ],
            }
        )
    return examples


def load_test_record(
    data_dir: str,
    ecg_id: int,
    *,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    bundle = load_ptbxl_bundle(data_dir)
    rows = bundle.metadata[
        (bundle.metadata["ecg_id"] == int(ecg_id)) & (bundle.metadata["split"] == "test")
    ]
    if len(rows) == 0:
        raise PTBXLDataError(f"ecg_id {ecg_id} is not a PTB-XL test-partition record")
    row = rows.iloc[0]
    position = int(bundle.metadata.index[bundle.metadata["ecg_id"] == int(ecg_id)][0])
    preprocessing = PreprocessingConfig()
    signal: Optional[np.ndarray] = None
    if cache_dir and os.path.isdir(cache_dir):
        cache, manifest = read_signal_cache(cache_dir)
        lookup = {int(value): index for index, value in enumerate(manifest.get("ecg_ids", []))}
        if int(ecg_id) in lookup:
            signal = np.asarray(cache[lookup[int(ecg_id)]], dtype=np.float32)
    if signal is None:
        signal = read_ptbxl_signal(
            data_dir,
            row["filename_lr"],
            sampling_rate=preprocessing.sampling_rate,
            expected_length=preprocessing.sequence_length,
        )
    labels = bundle.labels[position]
    mask = bundle.label_mask[position]
    info = {
        "ecg_id": int(ecg_id),
        "patient_id": int(row["patient_id"]),
        "fold": int(row["strat_fold"]),
        "true_categories": [
            category
            for index, category in enumerate(TARGET_CATEGORIES)
            if bool(mask[index]) and float(labels[index]) == 1.0
        ],
        "label_note": "Dataset label shown for evaluation context; it is not a model output.",
    }
    return signal, info
