# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent calibration fixtures and fail-closed benchmark admission."""

import copy
import csv
import hashlib
import json
import subprocess
import sys
from itertools import permutations
from pathlib import Path

import pytest

from aibrix_kvcache.compression_profiler.recorder import TraceRunStatus
from aibrix_kvcache.compression_profiler.schema import TraceEvent
from tools.compression_memory_sample import COLUMNS, MemorySampler
from tools.compression_profiler_calibrate import (
    CalibrationGateError,
    calibration_config,
    code_fingerprint,
    config_fingerprint,
    evaluate_calibration,
    run_calibration,
    validate_calibration,
)


def row(name, start, end, kind="codec_encode", **values):
    values.setdefault("result", "calibrated")
    return TraceEvent(
        experiment_id="calibration", run_id="run", request_id="request",
        operation_id=name, event_type=kind, start_ns=start, end_ns=end,
        **values,
    ).to_record()

def fixture():
    # The queue is 5 ms, CPU work 1 ms, with independent 5/10 ms branches.
    rows = [
        row("cpu", 5_000_000, 6_000_000, enqueue_ns=0),
        row("short", 0, 5_000_000, "l2_get"),
        row("long", 0, 10_000_000, "l2_get"),
        row("done", 10_000_000, 10_000_000, "first_token",
            dependency_ids=("cpu", "short", "long")),
    ]
    checks = {
        "cpu_1ms": dict(operation_id="cpu", metric="host_ns",
                        event_type="codec_encode", expected_ns=1_000_000,
                        tolerance_ns=100_000),
        "queue": dict(operation_id="cpu", metric="queue_ns",
                      expected_ns=5_000_000, tolerance_ns=100_000),
        "overlap": dict(operation_id="done", metric="critical_path_ns",
                        expected_ns=10_000_000, tolerance_ns=100_000),
    }
    return rows, checks

def test_calibration_checks_attribution_queue_and_parallel_critical_path():
    rows, checks = fixture()
    result = evaluate_calibration(rows, checks, TraceRunStatus())
    assert result["status"] == "PASS"
    assert result["checks"]["queue"]["observed_ns"] == 5_000_000
    assert result["checks"]["overlap"]["observed_ns"] == 10_000_000
    rows[0]["start_ns"] = 4_000_000
    result = evaluate_calibration(rows, checks, TraceRunStatus())
    assert result["status"] == "FAIL"

def test_wrong_stage_cannot_pass_matching_delay():
    rows, checks = fixture()
    rows[0]["event_type"] = "l2_get"
    report = evaluate_calibration(rows, checks, TraceRunStatus())
    assert report["status"] == "FAIL"
    assert report["checks"]["cpu_1ms"]["status"] == "FAIL"

@pytest.mark.parametrize("duration", [1_000_000, 5_000_000, 10_000_000])
def test_explicit_tolerance_boundary_and_unavailable_are_not_zero(duration):
    check = {"delay": dict(operation_id="cpu", metric="host_ns",
                           expected_ns=duration, tolerance_ns=50_000)}
    rows = [row("cpu", 0, duration + 50_000)]
    result = evaluate_calibration(rows, check, TraceRunStatus())
    assert result["status"] == "PASS"
    rows[0]["end_ns"] += 1
    result = evaluate_calibration(rows, check, TraceRunStatus())
    assert result["status"] == "FAIL"
    check["delay"]["available"] = False
    result = evaluate_calibration(rows, check, TraceRunStatus())
    assert result["status"] == "BLOCKED"
    assert result["checks"]["delay"]["observed_ns"] is None

@pytest.mark.parametrize("fault", ["dropped", "invalid", "bytes", "unknown"])
def test_dropped_invalid_and_unknown_byte_observations_never_pass(fault):
    rows, checks = fixture()
    status = TraceRunStatus()
    if fault == "dropped":
        status = TraceRunStatus(dropped_events=1)
    elif fault == "invalid":
        rows[0]["dependency_ids"] = ["missing"]
    else:
        rows[0].update(result="encoded", raw_bytes=5, payload_bytes=5,
                       metadata_bytes=2, framing_bytes=1, envelope_bytes=8)
        rows[0]["envelope_bytes"] = 9 if fault == "bytes" else None
    assert evaluate_calibration(rows, checks, status)["status"] != "PASS"

def test_gate_rejects_missing_stale_nonpass_and_malformed_artifacts(tmp_path):
    path = tmp_path / "calibration.json"
    with pytest.raises(CalibrationGateError):
        validate_calibration(path, code_hash="code", config_hash="config")
    good = admission_fixture(tmp_path)
    path.write_text(json.dumps(good))
    expected = config_fingerprint(calibration_config())
    assert validate_calibration(path, code_hash="code", config_hash=expected)
    for change in (
        {"code_hash": "stale"}, {"config_hash": "stale"},
        {"status": "BLOCKED"}, {"dropped_events": 1},
        {"checks": {}}, {"byte_validation": {"valid": True, "unknowns": ["x"]}},
    ):
        path.write_text(json.dumps(good | change))
        with pytest.raises(CalibrationGateError):
            validate_calibration(path, code_hash="code", config_hash=expected)
    for malformed in ("null", "[]", "{"):
        path.write_text(malformed)
        with pytest.raises(CalibrationGateError):
            validate_calibration(path, code_hash="code", config_hash=expected)

def admission_fixture(path=None):
    checks = {}
    for name, metric, duration in (
        ("cpu_1ms", "host_ns", 1_000_000),
        ("cpu_5ms", "host_ns", 5_000_000),
        ("cpu_10ms", "host_ns", 10_000_000),
        ("storage", "host_ns", 5_000_000),
        ("queue", "queue_ns", 5_000_000),
        ("parallel_overlap", "critical_path_ns", 10_000_000),
        ("cuda_kernel", "device_ns", 50_000),
        ("h2d", "device_ns", 100_000),
    ):
        checks[name] = dict(
            operation_id=("parallel_done" if name == "parallel_overlap"
                          else name),
            metric=metric,
            event_type={"storage": "l2_get", "h2d": "kv_import",
                        "parallel_overlap": "first_token"}.get(
                            name, "codec_encode"),
            expected_ns=duration, observed_ns=duration,
            tolerance_ns=max(1_000 if metric == "device_ns" else 250_000,
                             round(duration * 0.25)),
            available=True, status="PASS",
        )
    rows = [row(n, 0, d) for n, d in (
        ("cpu_1ms", 1_000_000), ("cpu_5ms", 5_000_000),
        ("cpu_10ms", 10_000_000))]
    rows += [row("storage", 0, 5_000_000, "l2_get"),
             row("queue", 5_000_000, 6_000_000, enqueue_ns=0),
             row("parallel_5ms", 0, 5_000_000, "l2_get"),
             row("parallel_10ms", 0, 10_000_000, "l2_get"),
             row("parallel_done", 10_000_000, 10_000_000, "first_token",
                 dependency_ids=("parallel_5ms", "parallel_10ms"))]
    checks["queue"].update(active_expected_ns=1_000_000,
                            active_observed_ns=1_000_000,
                            active_tolerance_ns=250_000)
    for name, duration, kind in (("cuda_kernel", 50_000, "codec_encode"),
                                  ("h2d", 100_000, "kv_import")):
        checks[name].update(reference_ns=[duration] * 3, expected_source=(
            "independent CUDA-event reference median"))
        for operation in [name] + [f"{name}_reference_{i}" for i in range(3)]:
            rows.append(row(operation, 0, duration, kind,
                            device_start=0, device_end=duration))
    checksum = hashlib.sha256(bytes(range(256)) * 16).hexdigest()
    for name, kind, result in (("identity_encode", "codec_encode", "encoded"),
                               ("identity_decode", "codec_decode", "decoded")):
        rows.append(row(name, 0, 1, kind, result=result, codec="identity",
                        raw_bytes=4096, payload_bytes=4096, metadata_bytes=2,
                        framing_bytes=1, envelope_bytes=4099,
                        checksum_algorithm="sha256", checksum_before=checksum,
                        checksum_after=checksum))
    contents = {"events.jsonl": "".join(json.dumps(r) + "\n" for r in rows),
                "run-status.json": json.dumps(dict(valid=True, reasons=[],
                                                   dropped_events=0))}
    if path:
        for name, content in contents.items():
            (path / name).write_text(content)
        sample = MemorySampler().sample()
        MemorySampler.write_csv(path / "memory.csv", [sample])
        for name in ("memory.csv", "memory.availability.json"):
            contents[name] = (path / name).read_bytes().decode()
    return dict(
        schema_version=1, status="PASS", code_hash="code",
        config_hash=config_fingerprint(calibration_config()),
        manifest={"config": calibration_config(), "time_unit": "ns",
                  "host_clock": "CLOCK_MONOTONIC_RAW",
                  "operations": sorted(r["operation_id"] for r in rows),
                  "memory_availability": ([sample["availability"]]
                                          if path else []),
                  "artifacts": {n: hashlib.sha256(c.encode()).hexdigest()
                                for n, c in contents.items()}},
        checks=checks, reasons=[], availability={n: True for n in checks},
        dropped_events=0,
        trace_validation=dict(valid=True, reasons=[]),
        byte_validation=dict(valid=True, reasons=[], unknowns=[]),
    )

@pytest.mark.parametrize("fault", ["operation_id", "gpu_reference",
    "gpu_availability", "top_availability", "events", "run_status", "summary",
    "missing_source", "duplicate", "failed_result", "gpu_device_tamper",
    "manifest_memory", "missing_gpu", "config", "tolerance", "expected",
    "gpu_expected", "empty_delays", "nan", "observed"])
def test_gate_recomputes_raw_bundle_and_rejects_tamper(tmp_path, fault):
    report = admission_fixture(tmp_path)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    assert validate_calibration(path, code_hash="code",
                                config_hash=report["config_hash"])
    if fault == "missing_gpu":
        del report["checks"]["cuda_kernel"]
    elif fault == "config":
        report["manifest"]["config"]["h2d_bytes"] = 12
    elif fault == "empty_delays":
        report["manifest"]["config"]["cpu_delays_ns"] = []
    elif fault in ("observed", "expected", "tolerance", "nan"):
        field, value = {"observed": ("observed_ns", 8_000_000),
            "expected": ("expected_ns", 123),
            "tolerance": ("tolerance_ns", 1_000_000_000),
            "nan": ("observed_ns", float("nan"))}[fault]
        report["checks"]["cpu_1ms"][field] = value
    elif fault == "operation_id":
        report["checks"]["cpu_1ms"]["operation_id"] = "h2d"
    elif fault == "gpu_reference":
        report["checks"]["cuda_kernel"]["reference_ns"] = [2, 3, 4]
    elif fault == "gpu_expected":
        report["checks"]["h2d"]["expected_ns"] = 123
    elif fault == "gpu_availability":
        report["checks"]["h2d"]["available"] = False
    elif fault == "top_availability":
        report["availability"]["h2d"] = False
    elif fault == "summary":
        report["manifest"]["operations"].pop()
    elif fault == "manifest_memory":
        report["manifest"]["memory_availability"] = [{"rss_bytes": True}]
    elif fault == "run_status":
        (tmp_path / "run-status.json").write_text(
            json.dumps(dict(valid=False, reasons=["dropped"],
                            dropped_events=1)))
    elif fault == "missing_source":
        (tmp_path / "events.jsonl").unlink()
    else:
        source = tmp_path / "events.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if fault == "duplicate":
            rows.append(rows[0])
        elif fault == "failed_result":
            rows[0]["result"] = "error"
        elif fault == "gpu_device_tamper":
            rows[8]["device_end"] = 1
        else:
            rows.pop()
        source.write_text("".join(json.dumps(r) + "\n" for r in rows))
        # Even rebinding the digest cannot hide invalid underlying evidence.
        report["manifest"]["artifacts"]["events.jsonl"] = hashlib.sha256(
            source.read_bytes()).hexdigest()
    path.write_text(json.dumps(report))
    with pytest.raises(CalibrationGateError):
        validate_calibration(path, code_hash="code",
                             config_hash=report["config_hash"])

def test_gate_stays_closed_with_python_optimization(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(admission_fixture() | {"status": "FAIL"}))
    code = (
        "from tools.compression_profiler_calibrate import "
        "validate_calibration; "
        f"validate_calibration({str(path)!r}, code_hash='code', "
        "config_hash='wrong')"
    )
    result = subprocess.run([sys.executable, "-O", "-c", code],
                             text=True, capture_output=True)
    assert result.returncode != 0
    assert "CalibrationGateError" in result.stderr

def test_fingerprints_bind_content_and_canonical_config(tmp_path):
    code = tmp_path / "code.py"
    code.write_text("version = 1")
    before = code_fingerprint([code])
    code.write_text("version = 2")
    assert code_fingerprint([code]) != before
    first = config_fingerprint({"a": 1, "b": 2})
    assert first == config_fingerprint({"b": 2, "a": 1})
    assert config_fingerprint({"a": 2}) != config_fingerprint({"a": 1})

def test_memory_columns_nulls_and_separate_logical_counters(tmp_path):
    sampler = MemorySampler(mode="light", providers={"l1_bytes": lambda: 100,
        "l2_bytes": lambda: 200, "codec_queue_depth": lambda: 3})
    sample = sampler.sample()
    assert sample["values"]["l1_bytes"] == 100
    assert sample["values"]["l2_bytes"] == 200
    assert sample["values"]["codec_queue_depth"] == 3
    for name, available in sample["availability"].items():
        assert available == (sample["values"][name] is not None)
    assert "total_bytes" not in sample["values"]
    path = tmp_path / "memory.csv"
    sampler.write_csv(path, [sample])
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(COLUMNS)
        output = next(reader)
    for name, available in sample["availability"].items():
        if not available:
            assert output[name] == "null"
    metadata = json.loads(path.with_suffix(".availability.json").read_text())
    assert metadata["samples"][0]["availability"] == sample["availability"]

@pytest.mark.parametrize("mode", ["light", "detailed"])
def test_sampling_only_snapshots_at_detailed_phase_boundaries(mode):
    # Use a real snapshot callback to observe the sampler's side effect.
    snapshots = []
    sampler = MemorySampler(mode=mode, snapshot=lambda: snapshots.append("s"))
    sampler.sample()
    assert snapshots == []
    sampler.sample(phase_boundary="warmup_end")
    assert snapshots == (["s"] if mode == "detailed" else [])

def test_counter_failure_unavailable_and_wrong_pid_hbm_cannot_be_relabelled():
    def broken():
        raise PermissionError("counter unavailable")

    sampler = MemorySampler(pid=99999999,
        providers={"l1_bytes": broken, "l2_bytes": lambda: -1})
    sample = sampler.sample()
    for field in ("rss_bytes", "pss_bytes", "l1_bytes", "l2_bytes",
                  "pinned_bytes", "hbm_allocated_bytes", "hbm_reserved_bytes"):
        assert sample["values"][field] is None
        assert sample["availability"][field] is False

def test_os_counter_units_and_missing_field_are_independent(tmp_path):
    proc = tmp_path / "1234"
    proc.mkdir()
    (proc / "status").write_text("VmRSS: 23 kB\nVmPin: 0 kB\n")
    (proc / "smaps_rollup").write_text("Pss: 17 kB\n")
    sampler = MemorySampler(pid=1234, proc_root=tmp_path)
    sample = sampler.sample()
    assert sample["values"]["rss_bytes"] == 23_552
    assert sample["values"]["pss_bytes"] == 17_408
    assert sample["values"]["pinned_bytes"] == 0
    assert sample["availability"]["pinned_bytes"] is True
    (proc / "status").write_text("VmRSS: 23 kB\n")
    assert sampler.sample()["values"]["pinned_bytes"] is None

def test_local_run_emits_manifest_events_and_no_false_cuda_pass(tmp_path):
    import torch
    config = calibration_config()
    report = run_calibration(tmp_path, config=config)
    assert (tmp_path / "calibration.json").exists()
    assert report["config_hash"] == config_fingerprint(config)
    assert report["manifest"]["time_unit"] == "ns"
    assert set((1_000_000, 5_000_000, 10_000_000)) <= {
        c.get("expected_ns") for c in report["checks"].values()
    }
    if not torch.cuda.is_available():
        assert report["status"] != "PASS"
        for name in ("cuda_kernel", "h2d"):
            assert report["checks"][name]["available"] is False
            assert report["checks"][name]["observed_ns"] is None
    with pytest.raises(CalibrationGateError):
        validate_calibration(tmp_path / "calibration.json",
                             code_hash="stale", config_hash="stale")

def test_cli_exit_nonzero_when_calibration_blocked(tmp_path):
    script = (Path(__file__).parents[1] /
              "tools/compression_profiler_calibrate.py")
    result = subprocess.run([sys.executable, str(script), "--output-dir",
                             str(tmp_path)], text=True, capture_output=True)
    report = json.loads((tmp_path / "calibration.json").read_text())
    assert result.returncode == (0 if report["status"] == "PASS" else 2)
    assert "status" in result.stdout

def test_invalid_configuration_rejected_before_artifacts(tmp_path):
    config = copy.deepcopy(calibration_config())
    config["tolerance_ns"] = -1
    with pytest.raises(ValueError):
        run_calibration(tmp_path, config=config)
    assert not (tmp_path / "calibration.json").exists()

def test_clock_failure_replaces_stale_pass_with_failed_artifact(
    tmp_path, monkeypatch
):
    import tools.compression_profiler_calibrate as calibrate

    def broken_clock():
        raise OSError("clock unavailable")

    (tmp_path / "calibration.json").write_text(json.dumps(admission_fixture()))
    (tmp_path / "events.jsonl").write_text("stale invalid trace\n")
    monkeypatch.setattr(calibrate, "raw_clock_ns", broken_clock)
    report = run_calibration(tmp_path)
    assert report["status"] == "FAIL"
    saved = json.loads((tmp_path / "calibration.json").read_text())
    assert saved["status"] == "FAIL"

@pytest.mark.parametrize("result", ["failed", "error", "cancelled", "rejected",
                                  "", "invented"])
def test_unsuccessful_stage_never_passes_timing(result):
    rows, checks = fixture()
    rows[0]["result"] = result
    assert evaluate_calibration(rows, checks, TraceRunStatus())["status"] == (
        "FAIL")

@pytest.mark.parametrize("delay", [1_000_000, 5_000_000, 10_000_000])
def test_cpu_thread_side_channel_cannot_hide_bad_emitted_duration(delay):
    checks = {"cpu": dict(operation_id="cpu", metric="cpu_active_ns",
                          expected_ns=delay, cpu_thread_ns=delay,
                          tolerance_ns=100_000)}
    rows = [row("cpu", 0, 1, result="calibrated")]
    assert evaluate_calibration(rows, checks, TraceRunStatus())["status"] == (
        "FAIL")

def test_queue_has_independently_checked_active_span():
    rows = [row("queue", 5_000_000, 50_000_000, enqueue_ns=0,
                result="calibrated")]
    checks = {"queue": dict(operation_id="queue", metric="queue_ns",
                            expected_ns=5_000_000, tolerance_ns=100_000)}
    assert evaluate_calibration(rows, checks, TraceRunStatus())["status"] == (
        "FAIL")

def test_torch_import_failure_keeps_complete_unavailable_device_checks(
        tmp_path, monkeypatch):
    from tools.compression_profiler_calibrate import _cuda_scenarios
    monkeypatch.setitem(sys.modules, "torch", None)
    checks = {}
    _cuda_scenarios(None, checks, calibration_config(), lambda: 0)
    assert set(checks) == {"cuda_kernel", "h2d"}
    for item in checks.values():
        assert item["available"] is False
        assert item["observed_ns"] is None
        assert item["status"] == "BLOCKED"
        assert "ModuleNotFoundError" in item["reason"]

@pytest.mark.parametrize("replace_index", [1, 2])
def test_memory_replace_failure_never_admits_mixed_pair(
        tmp_path, monkeypatch, replace_index):
    import tools.compression_memory_sample as memory
    path = tmp_path / "memory.csv"
    sampler = MemorySampler(providers={"l1_bytes": lambda: 10})
    sampler.write_csv(path, [sampler.sample()])
    old_csv = path.read_bytes()
    old_sidecar = path.with_suffix(".availability.json").read_bytes()
    replace = memory.os.replace
    attempts = []
    def failing_replace(source, destination):
        attempts.append(str(destination))
        if len(attempts) == replace_index:
            raise PermissionError("publication denied")
        replace(source, destination)
    monkeypatch.setattr(memory.os, "replace", failing_replace)
    with pytest.raises(PermissionError):
        sampler.write_csv(path, [sampler.sample()])
    if path.read_bytes() == old_csv:
        assert (path.with_suffix(".availability.json").read_bytes() ==
                old_sidecar)
        assert MemorySampler.read_csv(path)["rows"][0]["l1_bytes"] == 10
    else:
        with pytest.raises(ValueError):
            MemorySampler.read_csv(path)
    assert not list(tmp_path.glob(".*.tmp"))

def test_readonly_sidecar_is_replaced_as_a_consistent_generation(tmp_path):
    path = tmp_path / "memory.csv"
    sampler = MemorySampler(providers={"l1_bytes": lambda: 10})
    sampler.write_csv(path, [sampler.sample()])
    path.with_suffix(".availability.json").chmod(0o444)
    sampler.providers["l1_bytes"] = lambda: 20
    sampler.write_csv(path, [sampler.sample()])
    assert MemorySampler.read_csv(path)["rows"][0]["l1_bytes"] == 20

@pytest.mark.parametrize("order", list(permutations((
    "calibration.json", "memory.csv", "memory.availability.json"))))
@pytest.mark.parametrize("count", [1, 2, 3])
def test_gate_rejects_changing_bundle_then_accepts_committed_generation(
        tmp_path, monkeypatch, order, count):
    path = tmp_path / "calibration.json"
    before = admission_fixture(tmp_path)
    arguments = dict(code_hash="code", config_hash=before["config_hash"])
    path.write_text(json.dumps(before))
    following = tmp_path / "following"
    following.mkdir()
    after = admission_fixture(following)
    (following / path.name).write_text(json.dumps(after))
    read, published = Path.read_bytes, []
    def racing_read(source):
        content = read(source)
        if source == tmp_path / "memory.availability.json" and not published:
            for name in order[:count]:
                (following / name).replace(tmp_path / name)
            published.append(True)
        return content
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", racing_read)
        with pytest.raises(CalibrationGateError):
            validate_calibration(path, **arguments)
    assert published
    for name in order[count:]:
        (following / name).replace(tmp_path / name)
    accepted = validate_calibration(path, **arguments)
    assert accepted["manifest"] == after["manifest"]
