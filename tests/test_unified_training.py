"""Small CLI/config tests for the staged unified trainer.

Full training is an integration test that requires the real PTB-XL download;
these tests cover argument wiring and the stage-C defaults without touching a
dataset or running an epoch.
"""

from __future__ import annotations

import inspect
import json
from argparse import Namespace

from src.ptbxl_data import LEAD_ORDER, TARGET_CATEGORIES
from src.artifacts_helpers import build_stage_metadata
from src.train_unified_model import (
    build_parser,
    model_config_from_args,
    preprocessing_from_args,
    run_stage_c,
)


def test_preprocessing_is_ten_seconds_at_the_requested_rate():
    args = Namespace(sampling_rate=100, normalization="train_lead_zscore", clip_value=5.0)
    config = preprocessing_from_args(args)
    assert config.sampling_rate == 100
    assert config.sequence_length == 1000
    assert list(config.lead_order) == list(LEAD_ORDER)

    args = Namespace(sampling_rate=500, normalization="none", clip_value=4.0)
    assert preprocessing_from_args(args).sequence_length == 5000


def test_model_config_parses_channel_list():
    args = Namespace(
        sampling_rate=100,
        encoder_channels="8, 16, 32",
        classifier_hidden=24,
        dropout=0.1,
    )
    config = model_config_from_args(args)
    assert config.encoder_channels == (8, 16, 32)
    assert config.classifier_hidden == 24
    assert config.sequence_length == 1000


def test_cli_commands_and_all_mode_stage_c_source_default():
    parser = build_parser()
    args = parser.parse_args(["all"])
    # The "all" command does not define --stage-b; stage C must fall back to
    # its --out-dir rather than raising AttributeError.
    assert not hasattr(args, "stage_b")
    assert 'getattr(args, "stage_b", None)' in inspect.getsource(run_stage_c)

    for command in ("stage-a", "stage-b", "stage-c", "tune", "all"):
        parsed = parser.parse_args([command])
        assert parsed.command == command
        assert hasattr(parsed, "out_dir")
        assert hasattr(parsed, "data_dir")


def test_training_metadata_is_json_serializable_and_explicit():
    args = Namespace(sampling_rate=100, seed=7, batch_size=8, max_train_records=None, max_val_records=2)
    metadata = build_stage_metadata("C", args)
    json.dumps(metadata)
    assert metadata["training_stage"] == "C"
    assert metadata["run_settings"]["seed"] == 7
    assert metadata["run_settings"]["max_val_records"] == 2
    assert any("fold 9" in note for note in metadata["notes"])
    assert list(TARGET_CATEGORIES) == ["NORM", "MI", "STTC", "CD", "HYP"]
