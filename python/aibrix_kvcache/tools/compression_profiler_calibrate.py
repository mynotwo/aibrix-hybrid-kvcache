#!/usr/bin/env python3
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

"""RAW host spans, completed CUDA events and evidence-bound benchmark gates."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Source-tree CLI entry points must work from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aibrix_kvcache.compression.adapter import (
    IdentityCompressionAdapter,  # noqa: E402
)
from aibrix_kvcache.compression_profiler.recorder import (  # noqa: E402
    ShardedRingRecorder,
    TraceRunStatus,
)
from aibrix_kvcache.compression_profiler.schema import (  # noqa: E402
    EventType,
    TraceEvent,
)
from aibrix_kvcache.compression_profiler.validation import (  # noqa: E402
    validate_byte_accounting,
)

from tools import compression_memory_sample as memory  # noqa: E402
from tools.compression_impact_analyze import (  # noqa: E402
    TraceValidationError,
    build_operation_dag,
    critical_path,
)

ARTIFACTS = (
    "events.jsonl",
    "run-status.json",
    "memory.csv",
    "memory.availability.json",
)
IDENTITY_INPUT = bytes(range(256)) * 16


class CalibrationGateError(ValueError):
    """Formal benchmark admission failed; recalibrate this code/config."""


def calibration_config():
    return dict(
        cpu_delays_ns=[1_000_000, 5_000_000, 10_000_000],
        storage_delay_ns=5_000_000,
        queue_delay_ns=5_000_000,
        tolerance_ns=250_000,
        gpu_tolerance_ns=1_000,
        relative_tolerance=0.25,
        cuda_elements=1_048_576,
        h2d_bytes=4_194_304,
        cuda_reference_repeats=3,
    )


def config_fingerprint(config):
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def code_fingerprint(paths=None):
    if paths is None:
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "aibrix_kvcache").rglob("*.py"))
        paths += sorted((root / "tools").glob("compression_*.py"))
        paths = [(path.relative_to(root).as_posix(), path) for path in paths]
    else:
        paths = [(str(path), Path(path)) for path in paths]
    digest = hashlib.sha256()
    for name, path in sorted(paths):
        content = path.read_bytes()
        digest.update(name.encode() + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


def _within(check):
    numeric = all(
        type(check.get(k)) is int and check[k] >= 0
        for k in ("expected_ns", "observed_ns", "tolerance_ns")
    )
    return (
        check.get("available") is True
        and numeric
        and check["expected_ns"] > 0
        and check["observed_ns"] > 0
        and abs(check["observed_ns"] - check["expected_ns"])
        <= (check["tolerance_ns"])
    )


def evaluate_calibration(records, checks, status):
    records = list(records)
    byte_report = validate_byte_accounting(records)
    report = dict(
        status="FAIL",
        checks={},
        reasons=[],
        dropped_events=status.dropped_events,
        trace_validation={},
        byte_validation=asdict(byte_report),
    )
    try:
        dag = build_operation_dag(records, status)
        report["trace_validation"] = dict(valid=True, reasons=[])
    except TraceValidationError as exc:
        dag = None
        report["trace_validation"] = asdict(exc.report)
    success = {
        "calibrated",
        "encoded",
        "decoded",
        "stored",
        "loaded",
        "copied",
        "restored",
        "ok",
        "success",
        "hit",
    }
    for row in records:
        if row.get("result") not in success:
            report["reasons"].append(
                "unsuccessful_stage:" + row["operation_id"]
            )
    for name, spec in checks.items():
        value = dict(spec)
        value.setdefault("available", True)
        observed = None
        if value["available"] and dag is not None:
            row = dag.records.get(value.get("operation_id"))
            if (
                row is not None
                and row.get("result") in success
                and (
                    row.get("event_type")
                    == value.get("event_type", row.get("event_type"))
                )
            ):
                metric = value.get("metric")
                if metric in ("host_ns", "cpu_active_ns"):
                    observed = row["end_ns"] - row["start_ns"]
                elif metric == "queue_ns" and row.get("enqueue_ns") is not None:
                    observed = row["start_ns"] - row["enqueue_ns"]
                    value["active_observed_ns"] = (
                        row["end_ns"] - row["start_ns"]
                    )
                elif metric == "device_ns" and all(
                    row.get(k) is not None
                    for k in ("device_start", "device_end")
                ):
                    observed = round(row["device_end"] - row["device_start"])
                elif metric == "critical_path_ns":
                    operation = value["operation_id"]
                    observed = critical_path(dag, operation).total_ns
        value["observed_ns"] = observed
        active_ok = True
        if value.get("metric") == "queue_ns" and value["available"]:
            active_ok = _within(
                dict(
                    available=True,
                    expected_ns=value.get("active_expected_ns", 1_000_000),
                    observed_ns=value.get("active_observed_ns"),
                    tolerance_ns=value.get(
                        "active_tolerance_ns", value["tolerance_ns"]
                    ),
                )
            )
        value["status"] = (
            "BLOCKED"
            if not value["available"]
            else "PASS"
            if _within(value) and active_ok
            else "FAIL"
        )
        if value["status"] != "PASS":
            value["reason"] = value.get("reason") or (
                "missing/stage-mismatched observation"
                if observed is None
                else "active duration outside tolerance"
                if not active_ok
                else "duration outside tolerance"
            )
        report["checks"][name] = value
    invalid = (
        dag is None
        or not byte_report.valid
        or byte_report.unknowns
        or status.dropped_events != 0
        or not checks
        or report["reasons"]
        or any(c["status"] == "FAIL" for c in report["checks"].values())
    )
    blocked = any(c["status"] == "BLOCKED" for c in report["checks"].values())
    report["status"] = "FAIL" if invalid else "BLOCKED" if blocked else "PASS"
    return report


def validate_calibration(path, *, code_hash, config_hash):
    """Recompute evidence against CURRENT code and exact calibration config."""
    try:
        path = Path(path)
        sources, versions = memory.read_file_bundle(
            [path, *(path.parent / n for n in ARTIFACTS)]
        )
        report = json.loads(sources.pop(path.name))
        if report["status"] != "PASS":
            raise CalibrationGateError("nonPASS: " + str(report["status"]))
        config = report["manifest"]["config"]
        _validate_config(config)
        required = _required_checks(config)
        manifest, checks = report["manifest"], report["checks"]
        good = (
            type(report["schema_version"]) is int
            and report["schema_version"] == 1
            and report["status"] == "PASS"
            and report["code_hash"] == code_hash
            and report["config_hash"] == config_hash
            and config_fingerprint(config) == config_hash
            and manifest["time_unit"] == "ns"
            and manifest["host_clock"] == "CLOCK_MONOTONIC_RAW"
            and set(checks) == set(required)
            and report["availability"] == {n: True for n in required}
            and all(v is True for v in report["availability"].values())
        )
        good = good and manifest["artifacts"] == {
            n: hashlib.sha256(data).hexdigest() for n, data in sources.items()
        }
        pair = memory.decode_memory_pair(
            sources["memory.csv"], sources["memory.availability.json"]
        )
        good = good and manifest["memory_availability"] == [
            s["availability"] for s in pair["metadata"]["samples"]
        ]
        rows = [
            json.loads(line) for line in sources["events.jsonl"].splitlines()
        ]
        state = json.loads(sources["run-status.json"])
        if (
            set(state) != {"valid", "reasons", "dropped_events"}
            or type(state["valid"]) is not bool
            or type(state["dropped_events"]) is not int
            or not isinstance(state["reasons"], list)
            or any(type(r) is not str for r in state["reasons"])
        ):
            raise CalibrationGateError("invalid raw run status")
        observed = {r["operation_id"]: r for r in rows}
        ids = (
            {
                "identity_encode",
                "identity_decode",
                "parallel_5ms",
                "parallel_10ms",
                "parallel_done",
            }
            | (set(required) - {"parallel_overlap"})
            | {
                f"{n}_reference_{i}"
                for n in ("cuda_kernel", "h2d")
                for i in range(config["cuda_reference_repeats"])
            }
        )
        good = (
            good
            and len(rows) == len(ids)
            and set(observed) == ids
            and manifest["operations"] == sorted(ids)
        )
        for name, (metric, expected) in required.items():
            item = checks[name]
            operation = "parallel_done" if name == "parallel_overlap" else name
            if metric == "device_ns":
                refs = [
                    observed[f"{name}_reference_{i}"]
                    for i in range(config["cuda_reference_repeats"])
                ]
                reference = [
                    round(r["device_end"] - r["device_start"]) for r in refs
                ]
                expected = round(statistics.median(reference))
                good = (
                    good
                    and item["reference_ns"] == reference
                    and all(type(n) is int and n > 0 for n in reference)
                    and all(r["event_type"] == item["event_type"] for r in refs)
                    and item["expected_source"]
                    == "independent CUDA-event reference median"
                )
            canonical = _check(config, operation, metric, expected)
            good = (
                good
                and item["available"] is True
                and _within(item)
                and all(item.get(k) == v for k, v in canonical.items())
            )
        for name, expected in (
            ("parallel_5ms", 5_000_000),
            ("parallel_10ms", 10_000_000),
        ):
            r = observed[name]
            good = (
                good
                and r["event_type"] == "l2_get"
                and _within(
                    dict(
                        _check(config, name, "host_ns", expected),
                        available=True,
                        observed_ns=r["end_ns"] - r["start_ns"],
                    )
                )
            )
        for name, result in (
            ("identity_encode", "encoded"),
            ("identity_decode", "decoded"),
        ):
            r = observed[name]
            good = (
                good
                and r["result"] == result
                and r["codec"] == "identity"
                and r["event_type"] == name.replace("identity", "codec")
                and r["raw_bytes"] == len(IDENTITY_INPUT)
                and r["checksum_algorithm"] == "sha256"
                and r["checksum_before"]
                == hashlib.sha256(IDENTITY_INPUT).hexdigest()
                and r["checksum_before"] == r["checksum_after"]
            )
        recomputed = evaluate_calibration(rows, checks, TraceRunStatus(**state))
        good = (
            good
            and recomputed["status"] == "PASS"
            and all(
                config_fingerprint(report[k]) == config_fingerprint(v)
                for k, v in recomputed.items()
            )
        )
        if not good:
            raise CalibrationGateError("raw evidence/manifest mismatch/nonPASS")
        memory.verify_file_bundle(versions)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
    ) as exc:
        raise CalibrationGateError(f"calibration evidence: {exc}") from exc
    return report


def raw_clock_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _busy_ns(duration):
    start = raw_clock_ns()
    while raw_clock_ns() - start < duration:
        pass


def _check(config, operation, metric, expected, **extra):
    absolute = config[
        "gpu_tolerance_ns" if metric == "device_ns" else "tolerance_ns"
    ]
    tolerance = max(
        absolute, round((expected or 0) * config["relative_tolerance"])
    )
    stage = {
        "storage": "l2_get",
        "parallel_done": "first_token",
        "parallel_overlap": "first_token",
        "h2d": "kv_import",
    }.get(operation, "codec_encode")
    if metric == "queue_ns":
        extra |= dict(
            active_expected_ns=1_000_000,
            active_tolerance_ns=max(
                config["tolerance_ns"],
                round(1_000_000 * config["relative_tolerance"]),
            ),
        )
    return dict(
        operation_id=operation,
        metric=metric,
        expected_ns=expected,
        tolerance_ns=tolerance,
        event_type=stage,
        **extra,
    )


def _required_checks(config):
    return {
        f"cpu_{n // 1_000_000}ms": ("host_ns", n)
        for n in config["cpu_delays_ns"]
    } | {
        "storage": ("host_ns", config["storage_delay_ns"]),
        "queue": ("queue_ns", config["queue_delay_ns"]),
        "parallel_overlap": ("critical_path_ns", 10_000_000),
        "cuda_kernel": ("device_ns", None),
        "h2d": ("device_ns", None),
    }


def _validate_config(config):
    if set(config) != set(calibration_config()) or config["cpu_delays_ns"] != (
        [1_000_000, 5_000_000, 10_000_000]
    ):
        raise ValueError("full 1/5/10 ms calibration configuration required")
    for key, value in config.items():
        if key == "cpu_delays_ns":
            continue
        tolerance = key in (
            "tolerance_ns",
            "gpu_tolerance_ns",
            "relative_tolerance",
        )
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (value == 0 and not tolerance)
            or (key != "relative_tolerance" and type(value) is not int)
        ):
            raise ValueError("invalid calibration configuration: " + key)


def _host_scenarios(recorder, checks, config):
    def emit(name, kind, start, end, **values):
        recorder.record(
            TraceEvent(
                experiment_id="calibration",
                run_id="calibration",
                request_id="host",
                operation_id=name,
                event_type=kind,
                start_ns=start,
                end_ns=end,
                result="calibrated",
                **values,
            )
        )

    for delay in config["cpu_delays_ns"]:
        name = f"cpu_{delay // 1_000_000}ms"
        start, cpu = raw_clock_ns(), time.thread_time_ns()
        _busy_ns(delay)
        active, end = time.thread_time_ns() - cpu, raw_clock_ns()
        emit(name, "codec_encode", start, end, phase="cpu_busy")
        checks[name] = _check(
            config, name, "host_ns", delay, cpu_thread_ns=active
        )
    start = raw_clock_ns()
    time.sleep(config["storage_delay_ns"] / 1e9)
    emit("storage", "l2_get", start, raw_clock_ns(), phase="storage_wait")
    checks["storage"] = _check(
        config, "storage", "host_ns", config["storage_delay_ns"]
    )
    ready = threading.Event()

    def backlog():
        ready.wait()
        _busy_ns(config["queue_delay_ns"])

    def queued(enqueue):
        start = raw_clock_ns()
        _busy_ns(1_000_000)
        emit(
            "queue",
            "codec_encode",
            start,
            raw_clock_ns(),
            enqueue_ns=enqueue,
            phase="queued_busy",
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(backlog)
        enqueue = raw_clock_ns()
        second = executor.submit(queued, enqueue)
        ready.set()
        first.result()
        second.result()
    checks["queue"] = _check(
        config, "queue", "queue_ns", config["queue_delay_ns"]
    )
    barrier = threading.Barrier(3)
    bounds = []

    def branch(name, delay):
        barrier.wait()
        start = raw_clock_ns()
        time.sleep(delay / 1e9)
        end = raw_clock_ns()
        emit(name, "l2_get", start, end, phase="parallel_storage_wait")
        bounds.append((start, end))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(branch, name, delay)
            for name, delay in (
                ("parallel_5ms", 5_000_000),
                ("parallel_10ms", 10_000_000),
            )
        ]
        barrier.wait()
        for future in futures:
            future.result()
    end = max(b for _, b in bounds)
    emit(
        "parallel_done",
        "first_token",
        end,
        end,
        dependency_ids=("parallel_5ms", "parallel_10ms"),
    )
    checks["parallel_overlap"] = _check(
        config, "parallel_done", "critical_path_ns", 10_000_000
    )


def _identity(recorder, clock):
    adapter, raw = IdentityCompressionAdapter(), IDENTITY_INPUT
    start = clock()
    block = adapter.encode(
        raw,
        shape=(len(raw),),
        dtype="uint8",
        layout="flat",
        block_key="identity",
        generation=1,
        cache_group="g",
    )
    envelope = adapter.serialize(block)
    sizes = adapter.encoded_sizes(block, envelope)
    encoded_end = clock()
    decoded, end = adapter.decode(adapter.deserialize(envelope)), clock()
    common: dict[str, Any] = dict(
        experiment_id="calibration",
        run_id="calibration",
        request_id="identity",
        block_key="identity",
        generation=1,
        cache_group="g",
        cache_spec="flat",
        codec="identity",
        raw_bytes=len(decoded),
        payload_bytes=sizes.payload,
        metadata_bytes=sizes.metadata,
        framing_bytes=sizes.framing,
        envelope_bytes=sizes.envelope,
        checksum_algorithm="sha256",
        checksum_before=hashlib.sha256(raw).hexdigest(),
        checksum_after=hashlib.sha256(decoded).hexdigest(),
    )
    recorder.record(
        TraceEvent(
            **common,
            operation_id="identity_encode",
            event_type=EventType.CODEC_ENCODE,
            start_ns=start,
            end_ns=encoded_end,
            result="encoded",
        )
    )
    recorder.record(
        TraceEvent(
            **common,
            operation_id="identity_decode",
            event_type=EventType.CODEC_DECODE,
            start_ns=encoded_end,
            end_ns=end,
            result="decoded",
            dependency_ids=("identity_encode",),
        )
    )
    return decoded == raw


def _cuda_scenarios(recorder, checks, config, clock):
    for name in ("cuda_kernel", "h2d"):
        checks[name] = _check(
            config,
            name,
            "device_ns",
            None,
            available=False,
            observed_ns=None,
            status="BLOCKED",
            reason="CUDA unavailable",
        )
    try:
        import torch
    except ImportError as exc:
        for item in checks.values():
            if item["operation_id"] in ("cuda_kernel", "h2d"):
                item["reason"] = "torch unavailable: " + type(exc).__name__
        return
    if not torch.cuda.is_available():
        return
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        data = torch.zeros(config["cuda_elements"], device="cuda")
        host = torch.arange(config["h2d_bytes"], dtype=torch.int64)
        host = (host % 251).to(torch.uint8).pin_memory()
        target = torch.empty_like(host, device="cuda")
        stream.synchronize()

        for name, work in (
            ("cuda_kernel", lambda: data.add_(1)),
            ("h2d", lambda: target.copy_(host, non_blocking=True)),
        ):

            def timed(operation):
                start = clock()
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record(stream)
                work()
                b.record(stream)
                b.synchronize()
                duration, end = round(a.elapsed_time(b) * 1_000_000), clock()
                recorder.record(
                    TraceEvent(
                        experiment_id="calibration",
                        run_id="calibration",
                        request_id="cuda",
                        operation_id=operation,
                        event_type=(
                            EventType.CODEC_ENCODE
                            if name == "cuda_kernel"
                            else EventType.KV_IMPORT
                        ),
                        phase="fixed_cuda_workload",
                        result="calibrated",
                        start_ns=start,
                        end_ns=end,
                        device_start=0,
                        device_end=duration,
                    )
                )
                return duration

            work()
            stream.synchronize()
            reference = [
                timed(f"{name}_reference_{i}")
                for i in range(config["cuda_reference_repeats"])
            ]
            timed(name)
            checks[name] = _check(
                config,
                name,
                "device_ns",
                round(statistics.median(reference)),
                available=True,
                reference_ns=reference,
                expected_source="independent CUDA-event reference median",
            )
        if not torch.equal(target.cpu(), host) or not torch.all(
            data == 2 + config["cuda_reference_repeats"]
        ):
            recorder.invalidate("cuda_output_mismatch")


def run_calibration(output_dir, *, config=None):
    config = calibration_config() if config is None else dict(config)
    _validate_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must never leave an older PASS artifact admissible.
    (output / "calibration.json").write_text('{"status":"FAIL"}\n')
    (output / "events.jsonl").write_text("")
    available = hasattr(time, "CLOCK_MONOTONIC_RAW")
    clock = raw_clock_ns if available else time.monotonic_ns
    checks = {
        n: _check(
            config,
            "parallel_done" if n == "parallel_overlap" else n,
            metric,
            delay,
            available=False,
            observed_ns=None,
            status="BLOCKED",
            reason="scenario not executed",
        )
        for n, (metric, delay) in _required_checks(config).items()
    }
    sampler = memory.MemorySampler()
    samples = [sampler.sample()]
    recorder = ShardedRingRecorder(
        capacity=1024, output=output / "events.jsonl"
    )
    try:
        if not _identity(recorder, clock):
            recorder.invalidate("identity_bytes_mismatch")
        if available:
            _host_scenarios(recorder, checks, config)
        else:
            for name, (metric, delay) in _required_checks(config).items():
                checks[name] = _check(
                    config,
                    name,
                    metric,
                    delay,
                    available=False,
                    reason="CLOCK_MONOTONIC_RAW unavailable",
                )
        _cuda_scenarios(recorder, checks, config, clock)
    except Exception as exc:
        recorder.invalidate(type(exc).__name__)
    finally:
        recorder.close()
    samples.append(sampler.sample())
    sampler.write_csv(output / "memory.csv", samples)
    records = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text().splitlines()
    ]
    (output / "run-status.json").write_text(json.dumps(asdict(recorder.status)))
    report = evaluate_calibration(records, checks, recorder.status)
    report.update(
        schema_version=1,
        code_hash=code_fingerprint(),
        config_hash=config_fingerprint(config),
        availability={
            name: item["available"] for name, item in report["checks"].items()
        },
        manifest=dict(
            operations=sorted(r["operation_id"] for r in records),
            artifacts={
                n: hashlib.sha256((output / n).read_bytes()).hexdigest()
                for n in ARTIFACTS
            },
            platform=platform.platform(),
            python=platform.python_version(),
            config=config,
            time_unit="ns",
            host_clock=(
                "CLOCK_MONOTONIC_RAW"
                if available
                else "CLOCK_MONOTONIC (identity; host BLOCKED)"
            ),
            cpu_auxiliary_clock="thread_time_ns",
            cuda_clock="same-stream synchronized CUDA events",
            tolerance="abs(observed-expected) <= max(absolute ns, "
            "round(expected ns * relative_tolerance))",
            memory_availability=[s["availability"] for s in samples],
        ),
    )
    (output / "calibration.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    report = run_calibration(
        args.output_dir,
        config=(json.loads(args.config.read_text()) if args.config else None),
    )
    print(
        json.dumps(
            dict(
                status=report["status"],
                calibration=str(args.output_dir / "calibration.json"),
            )
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
