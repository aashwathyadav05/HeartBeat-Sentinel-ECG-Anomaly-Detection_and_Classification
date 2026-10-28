"""Loss-masking and batched-output tests for Stage B/C training runtime."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ptbxl_data import TARGET_CATEGORIES
from src.unified_ecg_model import UnifiedECGModel, UnifiedModelConfig
from src.unified_runtime import (
    class_pos_weight,
    combined_objective,
    masked_multilabel_bce_loss,
    masked_normal_reconstruction_loss,
    model_outputs_over_loader,
    normal_mask_from_labels,
)


def test_normal_mask_selects_only_usable_norm_only_records():
    labels = torch.tensor(
        [
            [1, 0, 0, 0, 0],  # NORM only -> normal
            [0, 1, 0, 0, 0],  # MI -> not normal
            [1, 1, 0, 0, 0],  # NORM + MI -> not normal
            [0, 0, 0, 0, 0],  # no mapped label -> unusable
            [1, 0, 0, 0, 0],  # NORM only but mask false -> unusable
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    result = normal_mask_from_labels(labels, mask)
    assert result.tolist() == [True, False, False, False, False]


def test_masked_multilabel_bce_ignores_invalid_entries():
    logits = torch.tensor([[0.2, -0.3, 2.5, -2.5, 9.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, False, False, False]])
    loss = masked_multilabel_bce_loss(logits, targets, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :2], targets[:, :2]
    )
    assert torch.allclose(loss, expected)

    # Masked-out extreme values cannot influence the loss.
    changed = logits.clone()
    changed[0, 2:] = -1000.0
    assert torch.allclose(
        masked_multilabel_bce_loss(changed, targets, mask), expected
    )


def test_masked_multilabel_bce_all_invalid_is_zero_with_grad():
    logits = torch.zeros(2, 5, requires_grad=True)
    targets = torch.ones(2, 5)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    loss = masked_multilabel_bce_loss(logits, targets, mask)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_masked_reconstruction_loss_excludes_abnormal_records():
    reconstruction = torch.zeros(3, 12, 4)
    target = torch.zeros(3, 12, 4)
    target[1] = 100.0  # abnormal record; must be ignored
    normal_mask = torch.tensor([True, False, True])
    loss = masked_normal_reconstruction_loss(reconstruction, target, normal_mask)
    expected = ((reconstruction[[0, 2]] - target[[0, 2]]) ** 2).mean()
    assert torch.allclose(loss, expected)
    assert loss.item() == pytest.approx(0.0)

    no_normal = masked_normal_reconstruction_loss(
        reconstruction, target, torch.tensor([False, False, False])
    )
    assert no_normal.item() == 0.0


def test_combined_objective_uses_lambda():
    cls_loss = torch.tensor(2.0)
    recon_loss = torch.tensor(4.0)
    combined = combined_objective(cls_loss, recon_loss, lambda_reconstruction=0.25)
    assert combined.item() == pytest.approx(3.0)


def test_class_pos_weight_is_clipped_and_safe():
    labels = np.asarray(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    mask = np.ones_like(labels, dtype=bool)
    weights = class_pos_weight(labels, mask)
    assert weights.shape == (5,)
    assert weights[0] == pytest.approx(1.0)  # 6 positives / 4 negatives clipped to 1
    assert weights[1] == pytest.approx(4.0)  # 2 positives / 8 negatives
    assert weights[2] == pytest.approx(1.0)  # no positives -> safe default
    assert weights[4] == pytest.approx(1.0)  # balanced
    assert np.all(weights >= 1.0)
    assert np.all(weights <= 20.0)


def test_model_outputs_over_loader_collects_aligned_arrays():
    torch.manual_seed(0)
    model = UnifiedECGModel(
        UnifiedModelConfig(
            sequence_length=1000,
            encoder_channels=(8, 16),
            decoder_channels=(8, 16),
            classifier_hidden=16,
            dropout=0.0,
        )
    ).eval()
    batch = {
        "signal": torch.randn(2, 12, 1000),
        "labels": torch.tensor(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32
        ),
        "label_mask": torch.ones(2, 5, dtype=torch.bool),
        "ecg_id": torch.tensor([11, 22]),
        "patient_id": torch.tensor([101, 102]),
        "fold": torch.tensor([1, 2]),
    }
    outputs = model_outputs_over_loader(model, [batch], torch.device("cpu"))

    assert outputs["errors"].shape == (2,)
    assert outputs["logits"].shape == (2, 5)
    assert outputs["scores"].shape == (2, 5)
    assert outputs["labels"].shape == (2, 5)
    assert outputs["label_mask"].shape == (2, 5)
    assert outputs["classification_loss"] is not None
    assert outputs["normal_reconstruction_loss"] is not None
    assert outputs["ecg_ids"] == [11, 22]
    assert outputs["patient_ids"] == [101, 102]
    assert outputs["folds"] == [1, 2]
    assert outputs["label_order"] == list(TARGET_CATEGORIES)
    assert np.all(np.isfinite(outputs["scores"]))
