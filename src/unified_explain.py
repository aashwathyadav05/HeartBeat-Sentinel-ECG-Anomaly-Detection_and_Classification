"""Explainability helpers for the unified PTB-XL model.

Occlusion-based temporal sensitivity is the documented method: a fixed-width
time window is replaced with the training-normalization mean (zero in
normalized space), and the change in a category logit is recorded. This is a
model-behaviour probe, not a physiological or causal explanation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .ptbxl_data import LEAD_ORDER, TARGET_CATEGORIES


def occlusion_sensitivity(
    model,
    normalized_signal: np.ndarray,
    category: str,
    *,
    window_samples: int = 100,
    stride_samples: int = 50,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return per-sample occlusion sensitivity for one category logit.

    Positive values mean that occluding the region *reduced* the target logit
    (the region supported the category under this probe).
    """
    import torch

    category = category.strip().upper()
    if category not in TARGET_CATEGORIES:
        raise ValueError(f"category must be one of {list(TARGET_CATEGORIES)}")
    signal = np.asarray(normalized_signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] != len(LEAD_ORDER):
        raise ValueError(f"Expected normalized signal shape (12, samples), got {signal.shape}")
    length = signal.shape[1]
    if window_samples < 1 or window_samples > length:
        raise ValueError("window_samples must be between 1 and the signal length")
    if stride_samples < 1:
        raise ValueError("stride_samples must be positive")
    if device is None:
        device = next(model.parameters()).device

    windows: List[tuple] = []
    starts = list(range(0, max(length - window_samples + 1, 1), stride_samples))
    if not starts or starts[-1] + window_samples < length:
        starts.append(max(length - window_samples, 0))
    for start in starts:
        end = min(start + window_samples, length)
        if (start, end) not in windows:
            windows.append((start, end))

    baseline = torch.from_numpy(signal).unsqueeze(0).to(device)
    occluded = np.repeat(signal[None, ...], len(windows), axis=0)
    for index, (start, end) in enumerate(windows):
        occluded[index, :, start:end] = 0.0
    batch = torch.from_numpy(np.ascontiguousarray(occluded)).to(device)
    category_index = list(TARGET_CATEGORIES).index(category)
    with torch.no_grad():
        baseline_logit = float(model(baseline).classification_logits[0, category_index].cpu())
        perturbed_logits = model(batch).classification_logits[:, category_index].cpu().numpy()
    deltas = baseline_logit - perturbed_logits

    saliency = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    region_rows = []
    for (start, end), delta in zip(windows, deltas):
        saliency[start:end] += float(delta)
        counts[start:end] += 1
        region_rows.append(
            {
                "start_sample": int(start),
                "end_sample": int(end),
                "start_seconds": float(start / 100.0),
                "end_seconds": float(end / 100.0),
                "logit_change": float(delta),
            }
        )
    counts[counts == 0] = 1.0
    saliency /= counts
    top = sorted(region_rows, key=lambda row: abs(row["logit_change"]), reverse=True)[:5]
    return {
        "method": "temporal_occlusion",
        "category": category,
        "target_logit": "category logit from the multilabel head",
        "baseline_logit": baseline_logit,
        "window_samples": int(window_samples),
        "stride_samples": int(stride_samples),
        "saliency": saliency.tolist(),
        "regions": region_rows,
        "top_regions": top,
        "limitations": [
            "Highlights model-sensitive time regions, not causal cardiac physiology.",
            "Sensitivity depends on window width and stride.",
            "A highlighted region does not establish or rule out a diagnosis.",
        ],
    }


def top_occlusion_regions(result: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    return list(result.get("top_regions", [])[:limit])
