"""Hand-calculated trace accounting and dependency-path regressions."""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aibrix_kvcache.compression_profiler.recorder import TraceRunStatus
from aibrix_kvcache.compression_profiler.schema import TraceEvent
from aibrix_kvcache.compression_profiler.validation import (
    validate_byte_accounting,
)
from tools.compression_impact_analyze import (
    TraceValidationError,
    build_operation_dag,
)
from tools.compression_impact_analyze import (
    critical_path as analyze_path,
)

from .test_compression_profiler_context import cache_factory as cache_factory
from .test_compression_profiler_context import records, worker_fixture
from .test_compression_profiler_context_failures import receive, seed


def critical_path(dag, terminal):
    result = analyze_path(dag, terminal)
    assert result.total_ns == result.active_ns + result.observed_queue_ns + (
        result.unobserved_ns)
    values = list(result.by_operation.values())
    assert sum(t.critical_ns for t in values) == (
        result.total_ns - result.unobserved_ns)
    assert 0 <= result.theoretical_lower_bound_ns <= result.total_ns
    assert 0 <= result.overlap_ns <= sum(t.active_ns for t in values)
    assert all(0 <= n <= result.total_ns for n in
               result.lower_bound_components_ns.values())
    for t in values:
        assert all(n is None or n >= 0 for n in (
            t.queue_ns, t.active_ns, t.overlap_ns, t.critical_ns, t.slack_ns
        ))
        assert t.overlap_ns <= t.active_ns
        assert t.critical_ns <= t.active_ns + (t.queue_ns or 0)
    return result


def event(name, start, end, kind="request", deps=(), **kwargs):
    return TraceEvent(
        experiment_id="experiment", run_id="run", request_id="request",
        operation_id=name, event_type=kind, start_ns=start, end_ns=end,
        dependency_ids=deps, **kwargs,
    ).to_record()


def parallel():
    return [event("load", 0, 2, "l2_get", result="loaded"),
        event("slow", 2, 10, "codec_decode", ("load",)),
        event("fast", 2, 6, "codec_decode", ("load",)),
        # Real cache_acquire encloses the reads it depends on.
        event("barrier", 0, 10, deps=("slow", "fast"), phase="cache_acquire"),
        event("import", 10, 12, "kv_import", ("barrier",)),
        event("first-token", 12, 12, "first_token", ("import",))]


def test_parallel_blocks_count_only_longest_branch_and_barrier_residual():
    result = critical_path(build_operation_dag(parallel()), "first-token")
    assert result.total_ns == 12
    assert result.by_operation["slow"].critical_ns == 8
    assert result.by_operation["fast"].critical_ns == 0
    assert result.by_operation["fast"].slack_ns == 4
    assert result.by_operation["barrier"].critical_ns == 0
    assert result.theoretical_lower_bound_ns == 8


def test_retry_fallback_shared_completed_work_counts_once():
    records = [event("load", 0, 3, "l2_get", result="load_error"),
        event("recovery", 0, 3, "error_recovery", ("load",)),
        event("retry", 3, 5, deps=("load", "recovery")),
        event("fallback", 5, 9, "fallback", ("load", "retry")),
        event("token", 9, 9, "first_token", ("fallback",))]
    result = critical_path(build_operation_dag(records), "token")
    assert result.total_ns == 9
    assert result.by_operation["load"].critical_ns == 3
    assert result.by_operation["recovery"].critical_ns == 0
    assert sum(x.critical_ns for x in result.by_operation.values()) == 9


def test_parent_ownership_span_is_not_repeated_as_child_work():
    records = [event("container", 0, 20),
        event("root", 0, 15, parent_operation_id="container"),
        event("load", 2, 8, "l2_get", parent_operation_id="root"),
        event("token", 9, 10, "first_token", ("load",),
              parent_operation_id="root")]
    result = critical_path(build_operation_dag(records), "token")
    assert result.total_ns == 10
    assert result.by_operation["root"].critical_ns == 3  # [0, 2] plus [8, 9].
    assert result.by_operation["load"].critical_ns == 6
    assert result.unobserved_ns == 0  # Ownership observed the post-load prefix.
    assert result.overlap_ns == 17  # Ignore parent overlap after this terminal.


@pytest.mark.parametrize("start,end", [(20, 30), (20, 20), (5, 30)])
def test_ownership_prefix_excludes_future_completion(start, end):
    rows = [event("future", start, end, "l2_get"),
            event("parent", 0, 40, deps=("future",)),
            event("token", 10, 12, "first_token", parent_operation_id="parent")]
    result = critical_path(build_operation_dag(rows), "token")
    assert "future" not in result.by_operation
    assert result.by_operation["parent"].critical_ns == 10
    assert result.active_ns == result.total_ns == 12
    rows[-1]["dependency_ids"] = ["future"]
    with pytest.raises(TraceValidationError,
                       match="dependency_completion_order"):
        build_operation_dag(rows)


def test_early_dependency_preserves_observed_parent_prefix():
    rows = [event("early", 0, 2), event("parent", 0, 20),
            event("token", 10, 12, "first_token", ("early",),
                  parent_operation_id="parent")]
    result = critical_path(build_operation_dag(rows), "token")
    assert result.by_operation["parent"].critical_ns == 8
    assert result.by_operation["early"].critical_ns == 2
    assert result.active_ns == result.total_ns == 12
    assert result.unobserved_ns == 0


def test_completed_parent_and_observed_queue_are_real_predecessor_costs():
    rows = [event("parent", 0, 10), event("other", 0, 2),
            event("child", 10, 12, deps=("other",),
                  parent_operation_id="parent")]
    result = critical_path(build_operation_dag(rows), "child")
    assert result.by_operation["parent"].critical_ns == 10
    rows = [event("load", 0, 1), event("token", 3, 4, deps=("load",),
                                      enqueue_ns=1)]
    result = critical_path(build_operation_dag(rows), "token")
    assert result.by_operation["load"].slack_ns == 0
    rows = [event("parent", 0, 5), event("child", 6, 7, enqueue_ns=2,
                                        parent_operation_id="parent")]
    result = critical_path(build_operation_dag(rows), "child")
    assert result.by_operation["parent"].slack_ns is None


def test_unrelated_same_request_time_neighbors_never_become_dependencies():
    records = [event("unrelated", 0, 100), event("token", 50, 52)]
    result = critical_path(build_operation_dag(records), "token")
    assert result.total_ns == 2
    assert set(result.by_operation) == {"token"}
    rows = [event("a", 0, 0), event("b", 5, 5)]
    rows.append(event("t", 5, 5, "first_token", ("a", "b")))
    result = critical_path(build_operation_dag(rows), "t")
    assert result.total_ns == result.unobserved_ns == 5


def test_queue_and_active_are_separate_and_null_is_not_zero():
    records = [event("load", 3, 8, "l2_get", enqueue_ns=1),
               event("token", 8, 9, "first_token", ("load",))]
    result = critical_path(build_operation_dag(records), "token")
    assert result.total_ns == 8
    assert result.by_operation["load"].queue_ns == 2
    assert result.by_operation["load"].active_ns == 5
    assert result.by_operation["token"].queue_ns is None
    assert result.queue_ns is None
    assert result.observed_queue_ns == 2
    assert result.active_ns == 6


def test_light_overlapping_windows_and_batch_spans_do_not_sum():
    records = [event("encode", 0, 8, "codec_encode"),
        event("put", 2, 10, "l2_put", ("encode",)),
        event("token", 10, 10, "first_token", ("put",))]
    result = critical_path(build_operation_dag(records), "token")
    assert result.total_ns == 10
    assert result.by_operation["put"].critical_ns == 2
    assert result.by_operation["put"].overlap_ns == 6
    assert result.theoretical_lower_bound_ns == 8


def test_lower_bound_keeps_full_resource_windows_but_only_unoverlapped_serial():
    rows = [event("codec", 0, 8, "codec_encode"),
            event("storage", 2, 12, "l2_put", ("codec",))]
    result = critical_path(build_operation_dag(rows), "storage")
    assert result.theoretical_lower_bound_ns == 10
    rows = [event("storage", 0, 10, "l2_get"),
            event("codec", 5, 7, "codec_decode"),
            event("serial", 7, 11, "decode", ("codec",)),
            event("done", 11, 11, "first_token", ("storage", "serial"))]
    result = critical_path(build_operation_dag(rows), "done")
    assert result.theoretical_lower_bound_ns == 11
    assert result.lower_bound_components_ns["serial"] == 1


@pytest.mark.parametrize("change,reason", [
        ({"dependency_ids": ["missing"]}, "missing_dependency"),
        ({"parent_operation_id": "missing"}, "missing_dependency"),
        ({"end_ns": None}, "unclosed_span"),
        ({"start_ns": -1}, "span_time"),
        ({"start_ns": 10, "end_ns": 2}, "span_order"),
        ({"experiment_id": ""}, "missing_experiment_id"),
        ({"run_id": None}, "missing_run_id"),
        ({"request_id": ""}, "missing_request_id"),
        ({"event_type": "invented"}, "event_type"),
        ({"block_key": "block", "generation": None}, "missing_generation"),
])
def test_invalid_trace_rejected_without_heuristic_repairs(change, reason):
    record = event("one", 0, 1)
    record.update(change)
    with pytest.raises(TraceValidationError, match=reason):
        build_operation_dag([record])


def test_parent_edges_participate_in_cycle_validation():
    records = [event("a", 0, 1, deps=("b",)),
               event("b", 0, 1, parent_operation_id="a")]
    with pytest.raises(TraceValidationError, match="cycle"):
        build_operation_dag(records)


@pytest.mark.parametrize("field", ["request_id", "run_id", "experiment_id"])
def test_cross_identity_dependency_is_invalid(field):
    records = [event("a", 0, 1), event("b", 1, 2, deps=("a",))]
    records[1][field] = "other"
    with pytest.raises(TraceValidationError, match="cross_identity"):
        build_operation_dag(records)


def test_duplicate_operation_invalid_and_bad_status_never_analyzed():
    record = event("one", 0, 1)
    with pytest.raises(TraceValidationError, match="duplicate_operation_id"):
        build_operation_dag([record, record])
    for status in (
        TraceRunStatus(False, ("export_error",)),
        TraceRunStatus(dropped_events=1),
    ):
        with pytest.raises(TraceValidationError):
            build_operation_dag([record], status)


def block(name, kind, deps=(), **values):
    return event(
        name, 0, 1, kind, deps, block_key="b", generation=1,
        cache_group="g", cache_spec="spec", **values,
    )


def test_shfs_accounting_includes_framing_and_decode_output():
    records = [
        block("encode", "codec_encode", raw_bytes=100, payload_bytes=40,
              metadata_bytes=20, framing_bytes=12, envelope_bytes=72,
              result="encoded"),
        block("store", "l2_put", ("encode",), connector_write_bytes=72,
              envelope_bytes=72, result="stored"),
        block("load", "l2_get", connector_read_bytes=72, result="loaded"),
        block("decode", "codec_decode", ("load",), raw_bytes=100,
              payload_bytes=40, metadata_bytes=20, framing_bytes=12,
              envelope_bytes=72, result="decoded"),
        block("restore", "kv_import", ("decode",), raw_bytes=100,
              result="restored", phase="restore"),
    ]
    assert validate_byte_accounting(records).valid
    records[-1]["raw_bytes"] = 99
    assert "decode_output_bytes" in validate_byte_accounting(records).reasons
    records[1]["connector_write_bytes"] = 71
    records[2]["connector_read_bytes"] = 71
    report = validate_byte_accounting(records)
    assert {"connector_write_bytes", "connector_read_bytes"} <= set(
        report.reasons)


def test_byte_nulls_remain_unknown_and_failed_events_skip_equations():
    record = block("one", "codec_decode", result="decoded", raw_bytes=100,
                   payload_bytes=40, metadata_bytes=20, framing_bytes=None,
                   envelope_bytes=72)
    report = validate_byte_accounting([record])
    assert report.valid
    assert "one:envelope_bytes" in report.unknowns
    record.update(result="rejected", framing_bytes=12, envelope_bytes=1)
    assert validate_byte_accounting([record]).valid


def test_envelope_and_negative_bytes_rejected():
    record = block("one", "codec_encode", result="encoded", payload_bytes=40,
                   metadata_bytes=20, framing_bytes=12, envelope_bytes=71)
    assert "envelope_bytes" in validate_byte_accounting([record]).reasons
    record["raw_bytes"] = -1
    assert "byte_value" in validate_byte_accounting([record]).reasons


def test_aggregate_closes_only_same_stage_direct_children():
    aggregate = event("batch", 0, 1, "codec_encode", result="encoded",
                      raw_bytes=200, envelope_bytes=144)
    records = [aggregate]
    for name in ("a", "b"):
        row = block(name, "codec_encode", result="encoded", raw_bytes=100,
                    envelope_bytes=72, parent_operation_id="batch")
        row["block_key"] = name
        records.append(row)
    assert validate_byte_accounting(records).valid
    records[0]["raw_bytes"] = 199
    assert "operation_raw_bytes" in validate_byte_accounting(records).reasons
    records[0]["raw_bytes"] = None
    report = validate_byte_accounting(records)
    assert report.valid
    assert "batch:operation_raw_bytes" in report.unknowns


def test_identity_conflict_is_not_hidden_by_equal_bytes():
    records = [block("load", "l2_get", result="loaded"),
               block("decode", "codec_decode", ("load",), result="decoded")]
    records[1]["generation"] = 2
    with pytest.raises(TraceValidationError, match="block_identity_conflict"):
        build_operation_dag(records)


def run_cli(tmp_path, records, status=None):
    source = tmp_path / "events.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    state = tmp_path / "run-status.json"
    state.write_text(json.dumps(status if status is not None else {
        "valid": True, "reasons": [], "dropped_events": 0,
    }))
    output = tmp_path / "analysis"
    script = Path(__file__).parents[1] / "tools/compression_impact_analyze.py"
    process = subprocess.run([
        sys.executable, str(script), "--events", str(source),
        "--run-status", str(state), "--output-dir", str(output),
    ], capture_output=True, text=True)
    return process, output


def test_cli_emits_stable_csv_and_json_with_null_observations(tmp_path):
    process, output = run_cli(tmp_path, parallel())
    assert process.returncode == 0, process.stderr
    report = json.loads((output / "critical_path.json").read_text())
    assert report["results"][0]["total_ns"] == 12
    assert report["results"][0]["by_operation"]["load"]["queue_ns"] is None
    with (output / "trace.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["queue_ns"] == ""
    assert rows[0]["raw_bytes"] == ""
    for name in ("request_summary.csv", "block_summary.csv"):
        assert (output / name).read_text().splitlines()[0]


@pytest.mark.parametrize("identity", ["request_id", "run_id", "experiment_id"])
def test_terminal_coverage_is_checked_per_identity(tmp_path, identity):
    extra = event("other", 0, 1)
    extra[identity] = "other"
    process, output = run_cli(tmp_path, parallel() + [extra])
    assert process.returncode == 2
    report = json.loads((output / "validation.json").read_text())
    assert not report["valid"] and "missing_terminal" in report["reasons"]
    extra["event_type"] = "output_token"
    assert run_cli(tmp_path, parallel() + [extra])[0].returncode == 0


@pytest.mark.parametrize("codec", [None, "IDENTITY"])
@pytest.mark.parametrize("kind,result,field,unknown", [
    ("codec_encode", "encoded", "raw_bytes", "raw_bytes"),
    ("codec_decode", "decoded", "raw_bytes", "decode_output_bytes"),
    ("l2_get", "loaded", "connector_read_bytes", "connector_read_bytes"),
    ("l2_put", "stored", "connector_write_bytes", "connector_write_bytes"),
])
def test_successful_missing_byte_observations_are_unknown(
    codec, kind, result, field, unknown
):
    row = block("one", kind, result=result, codec=codec, raw_bytes=100,
                payload_bytes=40, metadata_bytes=20, framing_bytes=12,
                envelope_bytes=72, connector_read_bytes=72,
                connector_write_bytes=72)
    row[field] = None
    report = validate_byte_accounting([row])
    assert report.valid and f"one:{unknown}" in report.unknowns
    for outcome in ("not_found", "load_error", "rejected"):
        row["result"] = outcome
        assert validate_byte_accounting([row]).unknowns == ()


@pytest.mark.parametrize("status", [
        {"valid": False, "reasons": ["export_error"], "dropped_events": 0},
        {"valid": True, "reasons": [], "dropped_events": 1},
        {},
        {"valid": "true", "reasons": [], "dropped_events": 0},
])
def test_cli_invalid_status_prevents_performance_claim(tmp_path, status):
    process, output = run_cli(tmp_path, parallel(), status)
    assert process.returncode == 2
    report = json.loads((output / "validation.json").read_text())
    assert not report["valid"]
    assert report["reasons"]
    assert not json.loads((output / "critical_path.json").read_text())["valid"]


def test_invalid_rerun_cannot_leave_valid_stale_conclusions(tmp_path):
    assert run_cli(tmp_path, parallel())[0].returncode == 0
    process, output = run_cli(tmp_path, [event("bad", 0, None)])
    assert process.returncode == 2
    assert not json.loads((output / "critical_path.json").read_text())["valid"]


def test_omitted_optional_times_do_not_crash_or_mean_zero():
    record = event("one", 2, 5, "first_token")
    del record["enqueue_ns"]
    result = critical_path(build_operation_dag([record]), "one")
    assert result.total_ns == 3
    assert result.by_operation["one"].queue_ns is None


def test_missing_status_export_error_counter_is_not_ignored(tmp_path):
    status = dict(valid=True, reasons=[], dropped_events=0,
                  export_error="disk full")
    process, output = run_cli(tmp_path, parallel(), status)
    assert process.returncode == 2
    report = json.loads((output / "validation.json").read_text())
    assert "export_error" in report["reasons"]


def test_explicit_counter_and_checksum_mismatches_are_invalid():
    row = block("one", "codec_decode", result="decoded", raw_bytes=100,
                checksum_algorithm="sha256", checksum_before="a",
                checksum_after="b")
    row.update(decode_output_bytes=99, connector_read_bytes=72,
               retrieved_object_bytes=71)
    report = validate_byte_accounting([row])
    assert {"decode_output_bytes", "connector_read_bytes", "checksum"} <= set(
        report.reasons)


def test_device_clock_is_never_subtracted_from_host_clock():
    row = event(
        "transfer", 100, 110, "kv_import", device_start=0.1, device_end=0.2
    )
    result = critical_path(build_operation_dag([row]), "transfer")
    assert result.total_ns == 10
    assert result.by_operation["transfer"].active_ns == 10


def test_multiple_terminals_keep_distinct_contributions_in_json(tmp_path):
    rows = [event("load", 0, 4, "l2_get"),
        event("first", 4, 5, "first_token", ("load",)),
        event("second", 5, 7, "output_token", ("first",))]
    process, output = run_cli(tmp_path, rows)
    assert process.returncode == 0, process.stderr
    values = json.loads((output / "critical_path.json").read_text())["results"]
    assert [r["total_ns"] for r in values] == [5, 7]
    assert [r["by_operation"]["load"]["critical_ns"] for r in values] == [4, 4]


@pytest.mark.parametrize("change,reason", [
    ({"generation": 1, "cache_group": "", "cache_spec": "s"},
     "missing_cache_group"),
    ({"generation": 1, "cache_group": "g", "cache_spec": ""},
     "missing_cache_spec"),
    ({"generation": 1, "cache_group": "g", "cache_spec": "s",
      "layer_observed": True, "layer": None}, "missing_layer"),
])
def test_unobserved_identity_fields_do_not_pass_as_named_blocks(change, reason):
    row = event("one", 0, 1, block_key="b")
    row.update(change)
    with pytest.raises(TraceValidationError, match=reason):
        build_operation_dag([row])


def test_shared_raw_batch_window_counts_once():
    rows = [event("a", 0, 8, "l2_get", phase="batch"),
        event("b", 0, 8, "l2_get", phase="batch"),
        event("done", 8, 9, "first_token", ("a", "b"))]
    result = critical_path(build_operation_dag(rows), "done")
    assert result.total_ns == 9
    assert sum(r.critical_ns for r in result.by_operation.values()) == 9
    assert result.overlap_ns == 8


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("codec", ["", "IDENTITY"])
def test_real_cache_acquire_spans_and_connector_parent_direction(
    cache_factory, monkeypatch, tmp_path, kind, codec
):
    if codec:
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND", "SHFS")
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_SHFS_ROOT", str(tmp_path / "l2"))
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_L2_CACHE_COMPRESSION", codec)
    cache, tokens, output = cache_factory(mode="LIGHT")
    seed(cache, tokens)
    worker, meta, _ = worker_fixture(kind, cache, tokens, monkeypatch)
    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    cache.trace_finish((meta.trace_context, start), "request")
    assert receive(worker, meta, kind) == len(tokens)
    rows = records(cache, output)
    dag = build_operation_dag(rows, cache._l2_cache._profiler_recorder.status)
    imported = [r for r in rows if r["phase"].startswith("connector_")]
    assert imported
    for row in imported:
        result = critical_path(dag, row["operation_id"])
        assert result.total_ns == row["end_ns"] - start


def test_cli_merges_process_files_and_analyzes_restore_terminals(tmp_path):
    """Scheduler and worker files; the restore waits for another request."""
    def of(request, record):
        return dict(record, request_id=request)

    def chain(request, *specs):
        return [
            of(request, event(name, start, end, kind, deps,
                              parent_operation_id=request, result=result))
            for name, start, end, kind, deps, result in specs
        ]

    scheduler = [
        of("cold", event("cold", 0, 1)),
        of("resumed", event("resumed", 5, 6)),
        *chain("resumed", ("evict", 5, 30, "eviction", ("put",),
                           "local_miss")),
    ]
    worker = [
        *chain("cold",
               ("export", 1, 3, "kv_export", (), "copied"),
               ("encode", 3, 25, "codec_encode", ("export",), "encoded"),
               ("put", 25, 28, "l2_put", ("encode",), "stored")),
        *chain("resumed",
               ("get", 30, 31, "l2_get", ("evict",), "loaded"),
               ("decode", 31, 35, "codec_decode", ("get",), "decoded"),
               ("import", 35, 36, "kv_import", ("decode",), "restored")),
    ]
    paths = []
    for name, rows in (("scheduler", scheduler), ("rank0", worker)):
        events = tmp_path / name / "events.jsonl"
        events.parent.mkdir()
        events.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (tmp_path / name / "run-status.json").write_text(json.dumps(
            {"valid": True, "reasons": [], "dropped_events": 0}))
        paths.append(tmp_path / name)
    output = tmp_path / "analysis"
    script = Path(__file__).parents[1] / "tools/compression_impact_analyze.py"
    process = subprocess.run([
        sys.executable, str(script),
        "--events", *(str(p / "events.jsonl") for p in paths),
        "--run-status", *(str(p / "run-status.json") for p in paths),
        "--terminal", "kv_import", "--output-dir", str(output),
    ], capture_output=True, text=True)

    assert process.returncode == 0, (
        process.stderr, (output / "validation.json").read_text())
    report = json.loads((output / "critical_path.json").read_text())
    (result,) = report["results"]
    # The restore's critical path reaches back into the cold request's save.
    assert {"encode", "evict", "decode"} <= set(result["by_operation"])


def test_every_events_file_needs_its_own_run_status(tmp_path):
    """A process that never closed its recorder has no run status."""
    from tools.compression_impact_analyze import analyze

    status = tmp_path / "run-status.json"
    status.write_text(json.dumps(
        {"valid": True, "reasons": [], "dropped_events": 0}))
    files = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in parallel()))
        files.append(path)

    assert analyze(files, [status], tmp_path / "out") == 2
    report = json.loads((tmp_path / "out" / "validation.json").read_text())
    assert "run_status" in report["reasons"]
