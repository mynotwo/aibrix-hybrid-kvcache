import copy
import json

import pytest

from aibrix_kvcache.compression_profiler.recorder import TraceRunStatus
from tools.qwen_next_acceptance import evaluate_run

RANKS = range(2)
BISECT = {"cold": [1, 2], "local_hit": [3, 4], "external": [3, 4]}
# Each stage runs after the one it consumes, as on a real server.
STAGES = ("kv_export", "codec_encode", "l2_put", "eviction", "l2_get",
          "codec_decode", "kv_import")


def _row(op, kind, request, deps=(), **values):
    return {
        "schema_version": 1,
        "experiment_id": "exp",
        "run_id": "run",
        "request_id": f"cmpl-bisect-{request}-0-abcd1234",
        "operation_id": op,
        "parent_operation_id": "",
        "dependency_ids": list(deps),
        "event_type": kind,
        "layer_observed": False,
        "start_ns": 10 * STAGES.index(kind),
        "end_ns": 10 * STAGES.index(kind) + 5,
        **values,
    }


def _profile():
    """Scheduler and two ranks: cold save, eviction, restore, verify."""
    puts = [f"put{r}" for r in RANKS]
    by_role = {
        "scheduler": [
            _row("evict", "eviction", "external", puts, result="local_miss")
        ]
    }
    for r in RANKS:
        by_role[f"rank{r}"] = [
            _row(f"export{r}", "kv_export", "cold", result="copied",
                 raw_bytes=100, transferred_bytes=100),
            _row(f"encode{r}", "codec_encode", "cold", [f"export{r}"],
                 result="encoded", raw_bytes=100, payload_bytes=30,
                 metadata_bytes=8, framing_bytes=2, envelope_bytes=40),
            _row(f"put{r}", "l2_put", "cold", [f"encode{r}"],
                 result="stored", envelope_bytes=40,
                 connector_write_bytes=40, stored_object_bytes=40),
            _row(f"get{r}", "l2_get", "external", ["evict"],
                 result="loaded", connector_read_bytes=40,
                 retrieved_object_bytes=40),
            _row(f"decode{r}", "codec_decode", "external", [f"get{r}"],
                 result="decoded", raw_bytes=100, decode_output_bytes=100,
                 payload_bytes=30, metadata_bytes=8, framing_bytes=2,
                 envelope_bytes=40),
            _row(f"import{r}", "kv_import", "external", [f"decode{r}"],
                 phase="restore", result="restored",
                 restore_result="readback_equal", raw_bytes=100,
                 transferred_bytes=100),
            _row(f"verify{r}", "kv_import", "local", phase="verify",
                 result="compared", restore_result="equal", raw_bytes=100),
        ]
    return by_role


def _evaluate(by_role=None, bisect=None):
    return evaluate_run(
        by_role or _profile(),
        TraceRunStatus(),
        bisect or BISECT,
        world_size=len(RANKS),
    )


def test_complete_chain_passes_and_reports_bytes():
    result = _evaluate()
    assert result["status"] == "PASS", result["failures"]
    assert result["raw_bytes_per_rank"] == 100
    assert result["stored_bytes_per_rank"] == 40
    assert result["compression_ratio"] == pytest.approx(2.5)


def test_benchmark_request_ids_are_recognized_by_acceptance(
    tmp_path, monkeypatch
):
    from tools.qwen_next_hybrid import ttft_lengths

    requests = []

    def ttft(base, ids, request_id):
        requests.append(request_id)
        return {"ttft_s": 0.1, "tokens": [3, 4]}

    monkeypatch.setattr(ttft_lengths, "prompt_ids", lambda *args: [1] * 401)
    monkeypatch.setattr(ttft_lengths, "ttft", ttft)
    monkeypatch.setattr(ttft_lengths, "post", lambda *args: {})
    monkeypatch.setattr(ttft_lengths.time, "sleep", lambda *args: None)
    ttft_lengths.main("8013", str(tmp_path), "1", "401")
    by_role = _profile()
    for rows in by_role.values():
        for row in rows:
            tag = row["request_id"].split("-")[2]
            request = next(r for r in requests if r == f"bisect-{tag}")
            row["request_id"] = f"cmpl-{request}-0-abcd1234"
    probe = json.loads((tmp_path / "s2-bisect.json").read_text())
    result = _evaluate(by_role, probe)
    assert result["status"] == "PASS", result["failures"]


def _drop(role, op):
    return lambda p, b: p[role].remove(
        next(r for r in p[role] if r["operation_id"] == op)
    )


def _set(role, op, **values):
    return lambda p, b: next(
        r for r in p[role] if r["operation_id"] == op
    ).update(values)


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (_drop("rank1", "put1"), "save"),
        (_set("rank0", "import0", result="error"), "load"),
        (_set("rank0", "get0", dependency_ids=[]), "profiler"),
        (_set("scheduler", "evict", result="inferred"), "eviction"),
        (_set("rank1", "verify1", restore_result="differs:1/5"), "bytewise"),
        (_set("rank0", "import0", restore_result="readback_differs"),
         "readback"),
        (_set("rank0", "encode0", envelope_bytes=41), "profiler"),
        (lambda p, b: b.update(external=[3, 5]), "token"),
        (lambda p, b: b.update(external=[], local_hit=[]), "token"),
    ],
)
def test_any_missing_or_inconsistent_evidence_fails(mutate, reason):
    by_role, bisect = _profile(), copy.deepcopy(BISECT)
    mutate(by_role, bisect)
    result = _evaluate(by_role, bisect)
    assert result["status"] == "FAIL"
    assert any(reason in f for f in result["failures"]), result["failures"]


def test_missing_verify_is_reported_not_silently_passed():
    by_role = _profile()
    by_role["rank0"] = [
        r for r in by_role["rank0"] if r.get("phase") != "verify"
    ]
    result = _evaluate(by_role)
    assert result["status"] == "FAIL"
    assert any("bytewise" in f for f in result["failures"])


def test_invalid_run_status_fails():
    result = evaluate_run(
        _profile(),
        TraceRunStatus(False, ("dropped_event",), 1),
        BISECT,
        world_size=len(RANKS),
    )
    assert result["status"] == "FAIL"


def test_status_is_never_complete():
    result = _evaluate()
    assert result["status"] in {"PASS", "FAIL"}
    assert json.dumps(result)


def _with_second_part(by_role):
    """Each rank also saved and restored a span part."""
    for r in RANKS:
        rows = by_role[f"rank{r}"]
        extra = []
        for row in rows:
            if row["request_id"].startswith("cmpl-bisect-local"):
                continue
            copy_ = dict(row, operation_id=row["operation_id"] + "-span")
            copy_["dependency_ids"] = [
                d + "-span" if d != "evict" else d
                for d in row["dependency_ids"]
            ]
            extra.append(copy_)
        rows.extend(extra)
    by_role["scheduler"][0]["dependency_ids"] += [
        f"put{r}-span" for r in RANKS
    ]
    return by_role


def test_a_record_of_several_parts_passes_and_sums_its_bytes():
    result = _evaluate(_with_second_part(_profile()))
    assert result["status"] == "PASS", result["failures"]
    assert result["raw_bytes_per_rank"] == 200
    assert result["stored_bytes_per_rank"] == 80


def test_any_failed_part_fails_the_run():
    by_role = _with_second_part(_profile())
    next(
        r for r in by_role["rank1"] if r["operation_id"] == "import1-span"
    ).update(result="error")
    result = _evaluate(by_role)
    assert result["status"] == "FAIL"
    assert any("load: rank 1" in f for f in result["failures"])


def test_ranks_with_different_part_counts_fail():
    by_role = _with_second_part(_profile())
    by_role["rank1"] = [
        r for r in by_role["rank1"]
        if not r["operation_id"].endswith("-span")
    ]
    by_role["scheduler"][0]["dependency_ids"].remove("put1-span")
    result = _evaluate(by_role)
    assert result["status"] == "FAIL"
    assert any("parts" in f for f in result["failures"]), result["failures"]
