"""Plots for the unified PTB-XL model.

All plotting functions return the saved path (when ``save_path`` is supplied)
and close their figures to avoid leaking resources in long-running servers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .ptbxl_data import LEAD_ORDER, TARGET_CATEGORIES

CATEGORY_COLORS = {
    "NORM": "#10b981",
    "MI": "#ef4444",
    "STTC": "#f59e0b",
    "CD": "#8b5cf6",
    "HYP": "#06b6d4",
}


def _finish(fig, save_path: Optional[str]):
    import matplotlib.pyplot as plt

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig


def select_lead(signal: np.ndarray, lead: str = "II") -> np.ndarray:
    array = np.asarray(signal)
    if lead not in LEAD_ORDER:
        raise ValueError(f"lead must be one of {list(LEAD_ORDER)}")
    if array.ndim != 2 or array.shape[0] != len(LEAD_ORDER):
        raise ValueError(f"Expected (12, samples) ECG, got {array.shape}")
    return array[LEAD_ORDER.index(lead)]


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    pointwise_error: np.ndarray,
    *,
    lead: str = "II",
    anomaly_score: Optional[float] = None,
    anomaly_threshold: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """Original vs reconstruction and point-wise squared error for one lead."""
    import matplotlib.pyplot as plt

    original_lead = select_lead(original, lead)
    reconstructed_lead = select_lead(reconstructed, lead)
    error_lead = select_lead(pointwise_error, lead)
    samples = np.arange(len(original_lead))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(samples, original_lead, label="Original", color="#3b82f6", linewidth=1.2)
    axes[0].plot(samples, reconstructed_lead, label="Reconstructed", color="#10b981", linewidth=1.2)
    title = f"Lead {lead}: original vs reconstruction"
    if anomaly_score is not None and anomaly_threshold is not None:
        status = "anomalous" if anomaly_score > anomaly_threshold else "normal"
        title += f" | error={anomaly_score:.4f} | threshold={anomaly_threshold:.4f} | {status}"
    axes[0].set_title(title)
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)
    axes[1].plot(samples, error_lead, color="#ef4444", linewidth=1.0)
    axes[1].fill_between(samples, error_lead, color="#ef4444", alpha=0.25)
    axes[1].set_title("Point-wise squared reconstruction error")
    axes[1].set_xlabel("Sample index (100 Hz)")
    axes[1].set_ylabel("Squared error")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_condition_scores(
    scores: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    save_path: Optional[str] = None,
):
    """Five independent sigmoid scores with their frozen thresholds."""
    import matplotlib.pyplot as plt

    categories = list(TARGET_CATEGORIES)
    values = [float(scores.get(category, 0.0)) for category in categories]
    cutoffs = [float(thresholds.get(category, 0.5)) for category in categories]
    colors = [
        CATEGORY_COLORS[category] if values[i] >= cutoffs[i] else "#64748b"
        for i, category in enumerate(categories)
    ]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(categories, values, color=colors, edgecolor="#0f172a")
    axis.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1, label="default 0.5")
    for bar, cutoff, value in zip(bars, cutoffs, values):
        axis.hlines(cutoff, bar.get_x() - 0.35, bar.get_x() + 0.35, color="#f8fafc", linewidth=2)
        axis.text(bar.get_x() + bar.get_width() / 2, min(value + 0.03, 0.98),
                  f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Independent sigmoid score")
    axis.set_title("Condition scores (white tick = frozen decision threshold)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_error_distribution(
    errors: np.ndarray,
    threshold: float,
    references: Optional[np.ndarray] = None,
    *,
    save_path: Optional[str] = None,
):
    """Reconstruction-error distribution with the frozen anomaly threshold."""
    import matplotlib.pyplot as plt

    errors = np.asarray(errors, dtype=float)
    fig, axis = plt.subplots(figsize=(9, 4.5))
    if references is not None:
        refs = np.asarray(references, dtype=float)
        valid = np.isfinite(refs)
        axis.hist(errors[valid & (refs == 0)], bins=40, alpha=0.65, label="Normal reference", color="#10b981")
        axis.hist(errors[valid & (refs == 1)], bins=40, alpha=0.65, label="Abnormal reference", color="#ef4444")
    else:
        axis.hist(errors, bins=40, alpha=0.7, label="Records", color="#3b82f6")
    axis.axvline(threshold, color="#f59e0b", linestyle="--", linewidth=2, label=f"Threshold {threshold:.4f}")
    axis.set_xlabel("Mean squared reconstruction error (normalized units)")
    axis.set_ylabel("Record count")
    axis.set_title("Reconstruction-error distribution")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_occlusion(
    saliency: Sequence[float],
    *,
    category: str,
    window_samples: int = 100,
    save_path: Optional[str] = None,
):
    """Temporal occlusion sensitivity for one category."""
    import matplotlib.pyplot as plt

    values = np.asarray(saliency, dtype=float)
    fig, axis = plt.subplots(figsize=(11, 3.5))
    axis.plot(np.arange(len(values)), values, color=CATEGORY_COLORS.get(category, "#3b82f6"))
    axis.fill_between(np.arange(len(values)), values, color=CATEGORY_COLORS.get(category, "#3b82f6"), alpha=0.25)
    axis.axhline(0, color="#94a3b8", linewidth=0.8)
    axis.set_title(f"{category} temporal occlusion sensitivity (window={window_samples} samples)")
    axis.set_xlabel("Sample index (100 Hz)")
    axis.set_ylabel("Logit change")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, save_path)
