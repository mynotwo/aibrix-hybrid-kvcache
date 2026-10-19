"""Admission must bind identity evidence to the known calibration input."""
import hashlib
import json

import pytest

from tests.test_compression_profiler_calibrate import admission_fixture
from tools.compression_profiler_calibrate import (
    CalibrationGateError,
    validate_calibration,
)


@pytest.mark.parametrize("fault", ["checksum", "length", "both"])
def test_gate_rejects_self_consistent_wrong_identity(tmp_path, fault):
    report = admission_fixture(tmp_path)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    args = dict(code_hash="code", config_hash=report["config_hash"])
    assert validate_calibration(path, **args)["status"] == "PASS"
    source = tmp_path / "events.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        if row["operation_id"] in ("identity_encode", "identity_decode"):
            if fault in ("checksum", "both"):
                row["checksum_before"] = row["checksum_after"] = "a" * 64
            if fault in ("length", "both"):
                row.update(raw_bytes=4, payload_bytes=4, envelope_bytes=7)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report["manifest"]["artifacts"]["events.jsonl"] = hashlib.sha256(
        source.read_bytes()).hexdigest()
    path.write_text(json.dumps(report))
    with pytest.raises(CalibrationGateError):
        validate_calibration(path, **args)
