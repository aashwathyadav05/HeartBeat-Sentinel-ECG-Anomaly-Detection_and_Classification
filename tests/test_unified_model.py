"""Unified shared-encoder model shape, multilabel and gradient tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ptbxl_data import TARGET_CATEGORIES
from src.unified_ecg_model import (
    MODEL_ARCHITECTURE,
    UnifiedECGModel,
    UnifiedModelConfig,
    count_parameters,
    create_model,
    model_config_from_checkpoint,
)


def _config(**overrides) -> UnifiedModelConfig:
    values = {
        "sequence_length": 1000,
        "encoder_channels": (8, 16),
        "decoder_channels": (8, 16),
        "classifier_hidden": 16,
        "dropout": 0.0,
    }
    values.update(overrides)
    return UnifiedModelConfig(**values)


def test_forward_shapes_and_multilabel_scores():
    model = UnifiedECGModel(_config()).eval()
    x = torch.randn(3, 12, 1000)
    output = model(x)

    assert output.reconstruction.shape == x.shape
    assert output.classification_logits.shape == (3, len(TARGET_CATEGORIES))
    assert output.classification_scores.shape == (3, len(TARGET_CATEGORIES))
    assert output.pooled_features.shape == (3, 2 * 16)
    assert output.features.shape[0] == 3
    assert torch.all(output.classification_scores >= 0)
    assert torch.all(output.classification_scores <= 1)
    # Independent sigmoid outputs, not a softmax distribution.
    assert not torch.allclose(
        output.classification_scores.sum(dim=1), torch.ones(3), atol=1e-4
    )
    assert torch.allclose(
        output.classification_scores,
        torch.sigmoid(output.classification_logits),
    )


def test_reconstruction_error_and_predict_scores():
    model = UnifiedECGModel(_config()).eval()
    x = torch.randn(4, 12, 1000)
    errors = model.reconstruction_error(x)
    assert errors.shape == (4,)
    assert torch.all(errors >= 0)
    pointwise = model.reconstruction_error(x, reduction="none")
    assert pointwise.shape == x.shape
    assert torch.allclose(errors, pointwise.mean(dim=(1, 2)))
    assert model.predict_scores(x).shape == (4, len(TARGET_CATEGORIES))
    with pytest.raises(ValueError):
        model.reconstruction_error(x, reduction="sum")


def test_input_contract_rejects_wrong_lead_count_or_rank():
    model = UnifiedECGModel(_config()).eval()
    with pytest.raises(ValueError, match="batch, 12, samples"):
        model(torch.randn(2, 11, 1000))
    with pytest.raises(ValueError, match="batch, 12, samples"):
        model(torch.randn(2, 12))
    with pytest.raises(ValueError, match="batch, 12, samples"):
        model(torch.randn(2, 12, 1000, 1))


def test_config_validation_and_roundtrip():
    with pytest.raises(ValueError):
        UnifiedModelConfig(input_channels=11)
    with pytest.raises(ValueError):
        UnifiedModelConfig(sequence_length=8)
    with pytest.raises(ValueError):
        UnifiedModelConfig(encoder_channels=())
    with pytest.raises(ValueError):
        # Decoder depth must match the four encoder stages.
        UnifiedModelConfig(decoder_channels=(8, 16, 32))
    with pytest.raises(ValueError):
        UnifiedModelConfig(dropout=1.0)

    config = _config()
    restored = UnifiedModelConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.feature_dim == 32
    assert create_model(config).architecture == MODEL_ARCHITECTURE


def test_encoder_freezing_and_task_parameter_groups():
    model = UnifiedECGModel(_config())
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    model.freeze_encoder(True)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    model.freeze_encoder(False)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())

    groups = model.task_parameters()
    assert set(groups) == {"encoder", "decoder", "classifier"}
    assert count_parameters(model, trainable_only=True) > 0


def test_both_heads_receive_gradients_through_shared_encoder():
    model = UnifiedECGModel(_config())
    x = torch.randn(2, 12, 1000)
    output = model(x)
    loss = (
        ((output.reconstruction - x) ** 2).mean()
        + torch.nn.functional.binary_cross_entropy_with_logits(
            output.classification_logits, torch.ones(2, 5)
        )
    )
    loss.backward()
    encoder_grads = [
        parameter.grad for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_grads
    assert all(torch.isfinite(grad).all() for grad in encoder_grads)
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    assert any(parameter.grad is not None for parameter in model.classifier.parameters())


def test_model_config_from_checkpoint_rejects_non_unified_checkpoint():
    with pytest.raises(ValueError, match="model_config"):
        model_config_from_checkpoint({"state_dict": {}})
    config = _config()
    restored = model_config_from_checkpoint({"model_config": config.to_dict()})
    assert restored == config
