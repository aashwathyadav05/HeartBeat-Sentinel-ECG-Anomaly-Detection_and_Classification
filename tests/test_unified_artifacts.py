"""Versioned artifact save/load, compatibility and threshold validation tests."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from src.ptbxl_data import LEAD_ORDER, TARGET_CATEGORIES
from src.unified_artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    artifact_paths,
    default_threshold_payload,
    load_unified_artifacts,
    read_artifact_summary,
    resolve_artifact_dir,
    save_unified_artifacts,
    write_latest_pointer,
    write_thresholds,
)
from src.unified_ecg_model import UnifiedModelConfig


def test_artifact_roundtrip_preserves_model_and_contract(synthetic_artifact):
    version_dir = synthetic_artifact["version_dir"]
    paths = artifact_paths(version_dir)
    for path in (paths.model_path, paths.metadata_path, paths.thresholds_path, paths.history_path):
        assert os.path.isfile(path)
    assert os.path.isfile(os.path.join(os.path.dirname(version_dir), "latest.json"))

    loaded = load_unified_artifacts(version_dir, device="cpu")
    assert loaded.paths.version_dir == os.path.abspath(version_dir)
    assert isinstance(loaded.model_config, UnifiedModelConfig)
    assert loaded.model_config.sequence_length == 1000
    assert loaded.model.training is False  # eval mode
    assert loaded.preprocessing.sequence_length == 1000
    assert list(loaded.preprocessing.lead_order) == list(LEAD_ORDER)
    assert loaded.normalization.mode == "train_lead_zscore"
    assert np.allclose(loaded.normalization.mean, 0.25)
    assert np.allclose(loaded.normalization.std, 2.0)
    assert loaded.metadata["label_order"] == list(TARGET_CATEGORIES)
    assert loaded.metadata["dataset"]["version"] == "1.0.3"
    assert loaded.metadata["training"]["stage"] == "B"
    assert "Educational research prototype" in loaded.metadata["disclaimer"]
    assert loaded.thresholds["label_order"] == list(TARGET_CATEGORIES)
    assert loaded.thresholds["anomaly_threshold"] == 0.5
    assert loaded.thresholds["class_thresholds"] == [0.5] * 5

    # State dict round-trip: same input produces exactly the same outputs.
    x = torch.randn(2, 12, 1000)
    source = synthetic_artifact["model"].eval()
    with torch.no_grad():
        expected = source(x)
        actual = loaded.model(x)
    assert torch.allclose(expected.reconstruction, actual.reconstruction)
    assert torch.allclose(expected.classification_logits, actual.classification_logits)


def test_artifact_summary_and_resolver_paths(synthetic_artifact, tmp_path):
    version_dir = synthetic_artifact["version_dir"]
    summary = read_artifact_summary(version_dir)
    assert summary["version_dir"] == os.path.abspath(version_dir)
    assert summary["label_order"] == list(TARGET_CATEGORIES)
    assert summary["anomaly_threshold"] == 0.5
    assert summary["class_thresholds"] == [0.5] * 5

    # A model.pt path resolves to its containing version directory.
    assert resolve_artifact_dir(os.path.join(version_dir, "model.pt")) == os.path.abspath(version_dir)
    # The artifact root resolves through latest.json.
    root = os.path.dirname(version_dir)
    assert resolve_artifact_dir(root) == os.path.abspath(version_dir)
    # A latest.json path resolves to the version directory.
    assert resolve_artifact_dir(os.path.join(root, "latest.json")) == os.path.abspath(version_dir)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArtifactNotFoundError):
        resolve_artifact_dir(str(empty))
    with pytest.raises(ArtifactNotFoundError):
        load_unified_artifacts(str(empty))


def test_missing_thresholds_make_artifact_incomplete(artifact_factory):
    artifact = artifact_factory(save_thresholds=False)
    with pytest.raises(ArtifactNotFoundError, match="thresholds.json"):
        load_unified_artifacts(artifact["version_dir"])


def test_incompatible_checkpoint_format_is_rejected(synthetic_artifact):
    paths = artifact_paths(synthetic_artifact["version_dir"])
    checkpoint = torch.load(paths.model_path, map_location="cpu", weights_only=True)
    checkpoint["artifact_format_version"] = 999
    torch.save(checkpoint, paths.model_path)
    with pytest.raises(ArtifactIncompatibleError, match="format"):
        load_unified_artifacts(synthetic_artifact["version_dir"])


def test_incompatible_label_order_is_rejected(synthetic_artifact):
    paths = artifact_paths(synthetic_artifact["version_dir"])
    with open(paths.metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["label_order"] = ["NORM", "MI", "STTC", "CD", "OTHER"]
    with open(paths.metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    with pytest.raises(ArtifactIncompatibleError, match="label order"):
        load_unified_artifacts(synthetic_artifact["version_dir"])


def test_incompatible_dataset_version_is_rejected(synthetic_artifact):
    paths = artifact_paths(synthetic_artifact["version_dir"])
    with open(paths.metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["dataset"]["version"] = "1.0.1"
    with open(paths.metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    with pytest.raises(ArtifactIncompatibleError, match="PTB-XL 1.0.1"):
        load_unified_artifacts(synthetic_artifact["version_dir"])


def test_threshold_payload_validation(synthetic_artifact):
    version_dir = synthetic_artifact["version_dir"]
    with pytest.raises(ArtifactIncompatibleError):
        write_thresholds(
            version_dir,
            {
                "label_order": ["NORM", "MI", "STTC", "CD", "OTHER"],
                "anomaly_threshold": 0.5,
                "class_thresholds": [0.5] * 5,
            },
        )
    with pytest.raises(ArtifactIncompatibleError):
        write_thresholds(
            version_dir,
            {
                "label_order": list(TARGET_CATEGORIES),
                "anomaly_threshold": -1.0,
                "class_thresholds": [0.5] * 5,
            },
        )
    with pytest.raises(ArtifactIncompatibleError):
        write_thresholds(
            version_dir,
            {
                "label_order": list(TARGET_CATEGORIES),
                "anomaly_threshold": 0.5,
                "class_thresholds": [0.5] * 4,
            },
        )
    payload = default_threshold_payload(0.25, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert payload["tuned_on"] == "PTB-XL validation fold 9 only"
    assert payload["frozen"] is False


def test_latest_pointer_and_multiple_versions(artifact_factory):
    first = artifact_factory(version="v1")
    second = artifact_factory(version="v2")
    root = os.path.dirname(first["version_dir"])
    pointer = write_latest_pointer(second["version_dir"])
    with open(pointer, encoding="utf-8") as handle:
        assert json.load(handle)["version_dir"] == "v2"
    assert resolve_artifact_dir(root) == os.path.abspath(second["version_dir"])
    # Without a pointer, the highest version wins.
    os.remove(pointer)
    assert resolve_artifact_dir(root) == os.path.abspath(second["version_dir"])


def test_save_requires_model_and_preprocessing_types(synthetic_artifact):
    # The loader is intentionally strict about a non-dict model_config.
    paths = artifact_paths(synthetic_artifact["version_dir"])
    checkpoint = torch.load(paths.model_path, map_location="cpu", weights_only=True)
    checkpoint["model_config"] = "not-a-config"
    torch.save(checkpoint, paths.model_path)
    with pytest.raises(ArtifactIncompatibleError, match="model configuration"):
        load_unified_artifacts(synthetic_artifact["version_dir"])
