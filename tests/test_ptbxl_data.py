"""PTB-XL metadata, labels, splits, validation and preprocessing tests.

These tests do not read any real PTB-XL waveform. They build tiny official-
shaped metadata/statement CSVs and synthetic 12x1000 arrays.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ptbxl_data import (
    LEAD_ORDER,
    TARGET_CATEGORIES,
    NormalizationStats,
    PatientLeakageError,
    PreprocessingConfig,
    PTBXLBundle,
    PTBXLDataError,
    PTBXLSignalDataset,
    PTBXLValidationError,
    anomaly_reference_from_labels,
    assign_split_from_fold,
    build_signal_cache,
    fit_lead_normalization,
    load_ptbxl_bundle,
    load_ptbxl_metadata,
    load_scp_statements,
    map_scp_codes_to_targets,
    normalize_signal,
    read_signal_cache,
    read_ptbxl_signal,
    select_frame,
    validate_ecg_signal,
    validate_patient_splits,
)


def _statement_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scp_code": "NORM", "diagnostic": 1, "diagnostic_class": "NORM"},
            {"scp_code": "IMI", "diagnostic": 1, "diagnostic_class": "MI"},
            {"scp_code": "LVH", "diagnostic": 1, "diagnostic_class": "HYP"},
            {"scp_code": "NDT", "diagnostic": 1, "diagnostic_class": "OTHER"},
            {"scp_code": "SR", "diagnostic": 0, "diagnostic_class": ""},
        ]
    ).set_index("scp_code")


def _write_ptbxl(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    rows = [
        # ecg_id, patient_id, fold, scp_codes
        (1, 101, 1, {"NORM": 100.0}),
        (2, 102, 2, {"IMI": 100.0}),
        (3, 103, 3, {"NORM": 100.0, "LVH": 80.0}),
        (4, 104, 9, {"NDT": 100.0}),  # diagnostic but outside five targets
        (5, 105, 10, {"NORM": 100.0}),
        (6, 106, 4, {"NORM": 50.0}),
        (7, 107, 5, {"NORM": 100.0, "IMI": 100.0}),
        (8, 108, 9, {"NORM": 100.0}),
        (9, 109, 10, {"IMI": 100.0}),
        (10, 110, 1, "not a dict"),  # unparseable -> excluded
        (11, 111, 99, {"NORM": 100.0}),  # unsupported fold -> dropped
        (12, 112, 1, {"NORM": 100.0, "SR": 100.0}),  # non-diagnostic ignored
    ]
    frame = pd.DataFrame(
        [
            {
                "ecg_id": ecg_id,
                "patient_id": patient_id,
                "strat_fold": fold,
                "scp_codes": codes if isinstance(codes, str) else str(codes),
                "filename_lr": f"records100/{ecg_id // 1000:05d}/{ecg_id:05d}_lr",
            }
            for ecg_id, patient_id, fold, codes in rows
        ]
    )
    frame.to_csv(os.path.join(data_dir, "ptbxl_database.csv"), index=False)
    _statement_frame().to_csv(
        os.path.join(data_dir, "scp_statements.csv"), index_label="scp_code"
    )


@pytest.fixture
def ptbxl_bundle(tmp_path):
    data_dir = str(tmp_path / "ptbxl")
    _write_ptbxl(data_dir)
    return load_ptbxl_bundle(data_dir), data_dir


def test_target_order_is_fixed_and_exact():
    assert TARGET_CATEGORIES == ("NORM", "MI", "STTC", "CD", "HYP")
    assert LEAD_ORDER == (
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    )


def test_metadata_load_and_unsupported_folds_are_dropped(tmp_path):
    data_dir = str(tmp_path / "ptbxl")
    _write_ptbxl(data_dir)
    metadata = load_ptbxl_metadata(data_dir)
    assert len(metadata) == 11
    assert 11 not in set(metadata["ecg_id"])
    assert metadata["ecg_id"].is_unique
    assert metadata.attrs["n_invalid_scp_codes"] == 1


def test_metadata_and_statement_files_are_required(tmp_path):
    with pytest.raises(Exception, match="metadata not found"):
        load_ptbxl_metadata(str(tmp_path))
    with pytest.raises(Exception, match="SCP statements file not found"):
        load_scp_statements(str(tmp_path))


def test_scp_mapping_multilabel_and_unknown_labels_are_masked():
    statements = _statement_frame()

    labels, matched, diagnostics, usable = map_scp_codes_to_targets(
        {"NORM": 100.0, "LVH": 90.0}, statements
    )
    assert usable is True
    assert labels.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert set(matched) == {"NORM", "LVH"}
    assert {code for code, _category, _likelihood in diagnostics} == {"NORM", "LVH"}

    # Diagnostic statement outside the five target classes is not a usable
    # five-way negative label set.
    labels, matched, diagnostics, usable = map_scp_codes_to_targets(
        {"NDT": 100.0}, statements
    )
    assert usable is False
    assert matched == []
    assert labels.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert diagnostics == [("NDT", "OTHER", 100.0)]

    # Non-diagnostic statements do not make a record usable.
    _labels, _matched, _diagnostics, usable = map_scp_codes_to_targets(
        {"SR": 100.0}, statements
    )
    assert usable is False

    # A mapped statement below the likelihood cut-off is excluded, not
    # silently treated as a negative.
    labels, matched, _diagnostics, usable = map_scp_codes_to_targets(
        {"NORM": 10.0}, statements, min_likelihood=50.0
    )
    assert usable is False
    assert matched == []
    assert labels.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_bundle_labels_masks_splits_and_anomaly_reference(ptbxl_bundle):
    bundle, _data_dir = ptbxl_bundle
    assert isinstance(bundle, PTBXLBundle)
    assert bundle.labels.shape == (11, 5)
    assert bundle.label_mask.shape == (11, 5)

    by_id = {int(row["ecg_id"]): position for position, row in bundle.metadata.iterrows()}
    # NORM-only record: normal reference.
    assert bundle.metadata.loc[by_id[1], "anomaly_reference"] == 0.0
    # MI and mixed NORM+HYP/MI are abnormal references.
    for ecg_id in (2, 3, 7, 9):
        assert bundle.metadata.loc[by_id[ecg_id], "anomaly_reference"] == 1.0
    # NORM plus a non-diagnostic rhythm statement stays normal.
    assert bundle.metadata.loc[by_id[12], "anomaly_reference"] == 0.0

    # Unknown diagnostic class and unparseable codes are excluded (all-False
    # mask), never all-negative targets.
    for ecg_id in (4, 10):
        position = by_id[ecg_id]
        assert not bundle.label_mask[position].any()
        assert np.isnan(bundle.metadata.loc[position, "anomaly_reference"])

    assert bundle.stats["n_records"] == 11
    assert bundle.stats["n_usable_for_supervision"] == 9
    assert bundle.stats["n_excluded_no_diagnostic_label"] == 2
    assert bundle.stats["positive_counts"] == {"NORM": 7, "MI": 3, "STTC": 0, "CD": 0, "HYP": 1}
    assert bundle.stats["target_order"] == list(TARGET_CATEGORIES)
    assert bundle.stats["split_summary"]["counts"] == {"train": 7, "val": 2, "test": 2}
    assert bundle.stats["split_summary"]["leakage"] == {}


def test_select_frame_uses_official_folds_and_normal_only(ptbxl_bundle):
    bundle, _data_dir = ptbxl_bundle
    frame, labels, mask = select_frame(
        bundle, "train", require_labels=True, normal_only=True
    )
    assert set(frame["ecg_id"]) == {1, 6, 12}
    assert labels.shape == (3, 5)
    assert mask.shape == (3, 5)
    assert assign_split_from_fold(1) == "train"
    assert assign_split_from_fold(8) == "train"
    assert assign_split_from_fold(9) == "val"
    assert assign_split_from_fold(10) == "test"
    assert assign_split_from_fold(0) is None
    assert assign_split_from_fold(11) is None


def test_patient_leakage_is_detected():
    metadata = pd.DataFrame({"patient_id": [1, 1, 2]})
    with pytest.raises(PatientLeakageError, match="Patient leakage"):
        validate_patient_splits(metadata, ["train", "val", "test"])
    summary = validate_patient_splits(
        metadata, ["train", "val", "test"], raise_on_leakage=False
    )
    assert "train-val" in summary["leakage"]


def test_anomaly_reference_never_infers_from_invalid_mask():
    labels = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    mask = np.asarray(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    references = anomaly_reference_from_labels(labels, mask)
    assert references[0] == 0.0
    assert references[1] == 1.0
    assert np.isnan(references[2])  # all-negative but no positive NORM
    assert np.isnan(references[3])  # unusable mask


def test_validate_ecg_signal_rejects_wrong_inputs_and_140_samples():
    valid = np.zeros((12, 1000), dtype=np.float32)
    result = validate_ecg_signal(valid)
    assert result.shape == (12, 1000)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]
    assert validate_ecg_signal(valid[None, ...]).shape == (12, 1000)

    with pytest.raises(PTBXLValidationError, match="ECG5000"):
        validate_ecg_signal(np.zeros(140, dtype=np.float32))
    with pytest.raises(PTBXLValidationError, match="exactly 12 leads"):
        validate_ecg_signal(np.zeros((11, 1000), dtype=np.float32))
    with pytest.raises(PTBXLValidationError, match="exactly 1000 samples"):
        validate_ecg_signal(np.zeros((12, 999), dtype=np.float32))
    with pytest.raises(PTBXLValidationError, match="NaN or infinite"):
        validate_ecg_signal(np.full((12, 1000), np.nan, dtype=np.float32))
    with pytest.raises(PTBXLValidationError, match="100 Hz"):
        validate_ecg_signal(valid, sampling_rate=500)
    with pytest.raises(PTBXLValidationError, match="numeric"):
        validate_ecg_signal(np.full((12, 1000), "x", dtype=object))


def test_preprocessing_config_and_normalization_contract():
    with pytest.raises(ValueError):
        PreprocessingConfig(sampling_rate=250)
    with pytest.raises(ValueError):
        PreprocessingConfig(lead_order=tuple(reversed(LEAD_ORDER)))

    stats = fit_lead_normalization([np.ones((12, 1000)), np.full((12, 1000), 3.0)])
    assert stats.mode == "train_lead_zscore"
    assert np.allclose(stats.mean, 2.0)
    assert np.allclose(stats.std, 1.0)

    normalized = normalize_signal(np.full((12, 1000), 2.0), stats)
    assert np.allclose(normalized, 0.0, atol=1e-6)

    clipped = normalize_signal(np.full((12, 1000), 100.0), stats)
    assert np.allclose(clipped, 5.0)

    per_record = NormalizationStats(mode="per_record_zscore", clip_value=5.0)
    signal = np.random.RandomState(1).randn(12, 1000).astype(np.float32)
    result = normalize_signal(signal, per_record)
    assert np.allclose(result.mean(axis=1), 0.0, atol=1e-5)
    assert np.allclose(result.std(axis=1), 1.0, atol=1e-4)

    none_stats = NormalizationStats(mode="none")
    assert np.allclose(normalize_signal(signal, none_stats), signal)


def test_signal_cache_skips_invalid_records_and_dataset_uses_cache(tmp_path):
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [10, 11, 12],
            "strat_fold": [1, 1, 1],
            "filename_lr": ["good1", "bad", "good3"],
        }
    )

    def reader(_data_dir, path, *, sampling_rate, expected_length):
        assert sampling_rate == 100
        assert expected_length == 1000
        if path == "bad":
            raise RuntimeError("synthetic unreadable record")
        value = 1.0 if path == "good1" else 3.0
        return np.full((12, 1000), value, dtype=np.float32)

    cache_dir = str(tmp_path / "cache")
    manifest = build_signal_cache(
        SimpleNamespace(metadata=metadata),
        str(tmp_path),
        cache_dir,
        record_reader=reader,
    )
    assert manifest["n_cached"] == 2
    assert manifest["n_skipped"] == 1
    assert manifest["ecg_ids"] == [1, 3]
    assert manifest["skipped"][0]["ecg_id"] == 2

    signals, loaded_manifest = read_signal_cache(cache_dir)
    assert signals.shape == (2, 12, 1000)
    assert loaded_manifest["sampling_rate"] == 100

    cached_metadata = metadata.iloc[[0, 2]].reset_index(drop=True)
    labels = np.zeros((2, 5), dtype=np.float32)
    labels[:, 0] = 1.0
    mask = np.ones((2, 5), dtype=bool)
    dataset = PTBXLSignalDataset(
        cached_metadata,
        labels,
        mask,
        str(tmp_path),
        PreprocessingConfig(),
        NormalizationStats(mode="none"),
        cache_dir=cache_dir,
        record_reader=reader,
    )
    item = dataset[0]
    assert item["signal"].shape == (12, 1000)
    assert item["ecg_id"] == 1
    assert item["labels"].shape == (5,)
    assert item["label_mask"].all()
    assert len(dataset) == 2
    assert len(dataset.dataset()) == 2

    with pytest.raises(PTBXLDataError, match="missing 1 requested record"):
        PTBXLSignalDataset(
            metadata,
            np.zeros((3, 5), dtype=np.float32),
            np.ones((3, 5), dtype=bool),
            str(tmp_path),
            PreprocessingConfig(),
            cache_dir=cache_dir,
            record_reader=reader,
        )


def test_signal_cache_fails_clearly_when_no_records_can_be_read(tmp_path):
    metadata = pd.DataFrame(
        {
            "ecg_id": [7],
            "patient_id": [70],
            "strat_fold": [1],
            "filename_lr": ["bad"],
        }
    )

    def reader(*_args, **_kwargs):
        raise RuntimeError("synthetic missing record")

    cache_dir = str(tmp_path / "empty-cache")
    with pytest.raises(PTBXLDataError, match="No signals were cached"):
        build_signal_cache(
            SimpleNamespace(metadata=metadata),
            str(tmp_path),
            cache_dir,
            record_reader=reader,
        )
    assert os.path.exists(os.path.join(cache_dir, "manifest.json"))
    with open(os.path.join(cache_dir, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["n_attempted"] == 1
    assert manifest["n_cached"] == 0
    assert manifest["n_failed"] == 1


def _mock_wfdb_record(tmp_path, names, monkeypatch):
    record_base = tmp_path / "record"
    record_base.with_suffix(".hea").write_text("synthetic WFDB header", encoding="utf-8")
    # Each source column has a distinct constant so the resulting channel order
    # can be checked without relying on a real waveform.
    p_signal = np.tile(np.arange(1, len(names) + 1, dtype=np.float32), (1000, 1))
    record = SimpleNamespace(p_signal=p_signal, fs=100, sig_name=names)
    fake_wfdb = SimpleNamespace(rdrecord=lambda _path: record)
    monkeypatch.setattr("src.ptbxl_data._import_wfdb", lambda: fake_wfdb)
    return record_base


@pytest.mark.parametrize(
    "names",
    [
        ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    ],
    ids=["official-uppercase", "mixed-case-aliases"],
)
def test_wfdb_lead_names_are_case_insensitive_and_output_is_canonical_order(
    tmp_path, monkeypatch, names
):
    # Deliberately scramble the columns: matching must canonicalize names and
    # restore the model's fixed lead order, not trust WFDB column order.
    names = [names[index] for index in (8, 5, 0, 6, 4, 1, 11, 3, 2, 7, 10, 9)]
    record_base = _mock_wfdb_record(tmp_path, names, monkeypatch)

    signal = read_ptbxl_signal(str(tmp_path), "record")

    assert signal.shape == (12, 1000)
    assert signal.dtype == np.float32
    assert np.isfinite(signal).all()
    canonical = {lead.casefold(): lead for lead in LEAD_ORDER}
    expected_source_values = [
        names.index(next(name for name in names if canonical.get(name.casefold()) == lead)) + 1
        for lead in LEAD_ORDER
    ]
    np.testing.assert_array_equal(signal[:, 0], expected_source_values)
    assert record_base.with_suffix(".hea").exists()


def test_wfdb_reader_reports_missing_leads(tmp_path, monkeypatch):
    names = [lead for lead in LEAD_ORDER if lead != "aVL"]
    _mock_wfdb_record(tmp_path, names, monkeypatch)

    with pytest.raises(PTBXLValidationError, match=r"missing=.*aVL"):
        read_ptbxl_signal(str(tmp_path), "record")


def test_wfdb_reader_reports_duplicate_leads_after_canonicalization(tmp_path, monkeypatch):
    names = list(LEAD_ORDER)
    names[names.index("aVL")] = "AVR"
    _mock_wfdb_record(tmp_path, names, monkeypatch)

    with pytest.raises(PTBXLValidationError, match=r"duplicate=.*aVR"):
        read_ptbxl_signal(str(tmp_path), "record")


def test_wfdb_reader_reports_unexpected_lead_names(tmp_path, monkeypatch):
    names = list(LEAD_ORDER)
    names[names.index("aVL")] = "X1"
    _mock_wfdb_record(tmp_path, names, monkeypatch)

    with pytest.raises(PTBXLValidationError, match=r"unexpected=.*X1"):
        read_ptbxl_signal(str(tmp_path), "record")
