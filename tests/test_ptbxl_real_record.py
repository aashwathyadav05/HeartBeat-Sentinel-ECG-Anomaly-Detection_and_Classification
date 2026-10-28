"""Optional smoke test against the official PTB-XL 100 Hz waveform."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.ptbxl_data import read_ptbxl_signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PTBXL_DIR = Path(
    os.environ.get("HEARTBEAT_PTBXL_DIR", str(PROJECT_ROOT / "data" / "ptbxl"))
)
RECORD_RELATIVE_PATH = Path("records100") / "00000" / "00001_lr"


@pytest.mark.skipif(
    not PTBXL_DIR.is_dir(),
    reason=f"PTB-XL dataset is absent at {PTBXL_DIR}; real-record smoke test skipped",
)
def test_official_ptbxl_100hz_record_loads_in_canonical_shape():
    record_path = PTBXL_DIR / RECORD_RELATIVE_PATH
    missing_files = [
        str(Path(str(record_path) + extension))
        for extension in (".hea", ".dat")
        if not Path(str(record_path) + extension).is_file()
    ]
    if missing_files:
        pytest.fail(
            "PTB-XL directory exists but the required smoke-test record is incomplete; "
            f"missing: {missing_files}"
        )

    signal = read_ptbxl_signal(
        str(PTBXL_DIR),
        str(RECORD_RELATIVE_PATH),
        sampling_rate=100,
        expected_length=1000,
    )

    assert signal.shape == (12, 1000)
    assert np.isfinite(signal).all()
