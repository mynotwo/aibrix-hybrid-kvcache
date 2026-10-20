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

# Copyright 2026 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from aibrix_kvcache.compression_profiler import recorder as recorder_module
from aibrix_kvcache.compression_profiler.recorder import (
    NullRecorder,
    ShardedRingRecorder,
    TraceRunStatus,
)
from aibrix_kvcache.compression_profiler.schema import EventType, TraceEvent
from aibrix_kvcache.compression_profiler.validation import validate_trace


def event(operation_id, **kwargs):
    return TraceEvent(
        "exp", "run", "req", operation_id, EventType.CODEC_ENCODE, **kwargs
    )


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_off_recorder_does_not_materialize_records():
    recorder = NullRecorder()
    assert recorder.record(lambda: pytest.fail("factory evaluated")) is False
    recorder.flush()
    recorder.close()


def test_close_atomically_publishes_real_run_status(tmp_path):
    output = tmp_path / "events.jsonl"
    recorder = ShardedRingRecorder(capacity=2, output=output)
    assert recorder.record(event("one"))
    recorder.invalidate("observation_error")
    recorder.close()

    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status == {
        "valid": False,
        "reasons": ["observation_error"],
        "dropped_events": 0,
    }


def test_second_recorder_owner_invalidates_shared_trace_run(tmp_path):
    output = tmp_path / "events.jsonl"
    owner = ShardedRingRecorder(capacity=2, output=output)
    duplicate = ShardedRingRecorder(capacity=2, output=output)
    assert not duplicate.record(event("must-not-export"))
    assert owner.record(event("owner-event"))
    duplicate.close()
    owner.close()
    failed_status = json.loads((tmp_path / "run-status.json").read_text())
    assert failed_status["valid"] is False
    assert "multiple_recorder_owners" in failed_status["reasons"]
    late_duplicate = ShardedRingRecorder(capacity=2, output=output)
    assert late_duplicate.record(event("clean-reopen"))
    late_duplicate.close()

    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status["valid"] is True
    assert [row["operation_id"] for row in read_records(output)] == [
        "clean-reopen"
    ]


def test_invalidation_is_thread_safe_idempotent_and_visible(tmp_path):
    NullRecorder().invalidate("observation_error")
    recorder = ShardedRingRecorder(capacity=16, output=tmp_path / "trace.jsonl")
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(recorder.invalidate, ["observation_error"] * 100))
        assert recorder.status == TraceRunStatus(
            False, ("observation_error",), 0
        )
        assert recorder.record(event("after-invalidation"))
        recorder.flush()
        assert not recorder.status.valid
        assert len(read_records(tmp_path / "trace.jsonl")) == 1
    finally:
        recorder.close()


def test_inflight_ring_full_never_overwrites_or_materializes(tmp_path,
                                                           monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    real_write = os.write

    def blocked_write(fd, data):
        entered.set()
        assert release.wait(5)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", blocked_write)
    output = tmp_path / "events.jsonl"
    recorder = ShardedRingRecorder(capacity=1, output=output)
    try:
        assert recorder.record(event("one"))
        assert entered.wait(5)
        assert not recorder.record(
            lambda: pytest.fail("full ring evaluated factory")
        )
        assert recorder.status.dropped_events == 1
        assert not recorder.status.valid
        assert "dropped_event" in recorder.status.reasons
    finally:
        release.set()
        recorder.close()
    assert [r["operation_id"] for r in read_records(output)] == ["one"]


def test_export_failure_is_fail_open_and_retains_unexported(tmp_path):
    recorder = ShardedRingRecorder(capacity=1, output=tmp_path)
    assert recorder.record(event("one"))
    recorder.flush()
    assert "export_error" in recorder.status.reasons
    assert not recorder.status.valid
    assert not recorder.record(event("two"))
    assert recorder.status.dropped_events == 1
    recorder.close()


def test_concurrent_shards_export_each_operation_once(tmp_path):
    output = tmp_path / "events.jsonl"
    recorder = ShardedRingRecorder(capacity=100, output=output)
    barrier = threading.Barrier(4)

    def produce(worker):
        barrier.wait(timeout=5)
        return [recorder.record(event(f"{worker}-{i}")) for i in range(100)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        accepted = list(executor.map(produce, range(4)))
    recorder.close()
    assert all(all(worker) for worker in accepted)
    ids = [r["operation_id"] for r in read_records(output)]
    assert len(ids) == 400
    assert set(ids) == {f"{w}-{i}" for w in range(4) for i in range(100)}
    assert recorder.status.valid


def test_json_and_file_io_are_background_only_batched_and_private(tmp_path,
                                                               monkeypatch):
    caller = threading.get_ident()
    writes = []
    opens = []
    first_write = threading.Event()
    release_write = threading.Event()
    real_write, real_open, real_dumps = os.write, os.open, json.dumps

    def write(fd, data):
        writes.append((threading.get_ident(), data))
        if len(writes) == 1:
            first_write.set()
            assert release_write.wait(5)
        return real_write(fd, data)

    def open_file(path, *args, **kwargs):
        if not isinstance(path, int):
            opens.append((threading.get_ident(), os.fspath(path)))
        return real_open(path, *args, **kwargs)

    def dumps(*args, **kwargs):
        assert threading.get_ident() != caller
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(json, "dumps", dumps)
    output = tmp_path / "events.jsonl"
    output.write_text("old contents")
    output.chmod(0o644)
    recorder = ShardedRingRecorder(capacity=20, output=output)
    try:
        assert recorder.record(event("0", start_ns=0, end_ns=1))
        assert first_write.wait(5)
        for i in range(1, 20):
            assert recorder.record(event(str(i), start_ns=0, end_ns=1))
    finally:
        release_write.set()
        recorder.close()
    assert opens
    assert all(
        thread != caller or path in (f"{output}.owner", f"{output}.owner.guard")
        for thread, path in opens
    )
    assert all(thread != caller for thread, _ in writes)
    assert [data.count(b"\n") for _, data in writes] == [1, 19]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert len(read_records(output)) == 20
    recorder.close()
    assert not recorder.record(lambda: pytest.fail("closed factory evaluated"))


def test_partial_write_invalidates_without_releasing_capacity(tmp_path,
                                                            monkeypatch):
    real_write = os.write

    def partial_write(fd, data):
        return real_write(fd, data[:5])

    monkeypatch.setattr(os, "write", partial_write)
    recorder = ShardedRingRecorder(capacity=1, output=tmp_path / "e.jsonl")
    assert recorder.record(event("one"))
    recorder.flush()
    assert not recorder.status.valid
    assert "export_error" in recorder.status.reasons
    assert not recorder.record(event("two"))
    recorder.close()


def test_bad_factory_is_fail_open_and_does_not_poison_other_events(tmp_path):
    def fail():
        raise ValueError("unavailable observation")

    recorder = ShardedRingRecorder(capacity=2, output=tmp_path / "e.jsonl")
    assert not recorder.record(fail)
    assert not recorder.record(object())
    assert recorder.record(event("good"))
    recorder.close()
    assert recorder.status.dropped_events == 2
    assert "record_error" in recorder.status.reasons
    assert read_records(tmp_path / "e.jsonl")[0]["operation_id"] == "good"


def test_flush_allows_reuse_and_close_joins_writer(tmp_path):
    output = tmp_path / "e.jsonl"
    recorder = ShardedRingRecorder(capacity=1, output=output)
    assert recorder.record(event("one"))
    recorder.flush()
    assert [r["operation_id"] for r in read_records(output)] == ["one"]
    assert recorder.record(event("two"))
    recorder.close()
    assert [r["operation_id"] for r in read_records(output)] == ["one", "two"]
    assert not any(t.name.startswith("compression-trace-")
                   for t in threading.enumerate())


def test_flush_barrier_does_not_count_later_events_from_an_earlier_shard(
    tmp_path, monkeypatch
):
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    b_write_started = threading.Event()
    release_b_write = threading.Event()
    original_shard = recorder_module._Shard
    real_write = os.write

    class SnapshotGate:
        def __init__(self):
            self.lock = threading.Lock()
            self.gated = False

        def __enter__(self):
            if threading.current_thread().name.startswith(
                "compression-trace-"
            ) and not self.gated:
                self.gated = True
                snapshot_started.set()
                assert release_snapshot.wait(5)
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    a_shard = original_shard(lock=SnapshotGate())
    shards = iter((a_shard, original_shard()))
    monkeypatch.setattr(recorder_module, "_Shard", lambda: next(shards))

    def write(fd, data):
        if b'"B-before"' in data:
            b_write_started.set()
            assert release_b_write.wait(5)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", write)
    output = tmp_path / "events.jsonl"
    recorder = ShardedRingRecorder(capacity=4, output=output)
    names = ("flush-one", "flush-two", "close")
    waiting = {name: threading.Event() for name in names}
    rechecked = {name: threading.Event() for name in names}
    probe_barriers = threading.Event()
    decisions = {name: [] for name in names}
    finished = {name: threading.Event() for name in names}
    real_wait_for = recorder._condition.wait_for

    def wait_for(predicate, timeout=None):
        name = threading.current_thread().name
        if name not in decisions:
            return real_wait_for(predicate, timeout)

        def observed():
            result = predicate()
            decisions[name].append(result)
            if len(decisions[name]) == 1:
                waiting[name].set()
            if probe_barriers.is_set():
                rechecked[name].set()
            return result

        return real_wait_for(observed, timeout)

    monkeypatch.setattr(recorder._condition, "wait_for", wait_for)

    def actor(name):
        (recorder.close if name == "close" else recorder.flush)()
        finished[name].set()

    threads = []
    try:
        assert recorder.record(event("A-before"))
        # Exporter has captured only shard A, but has not snapshotted its queue.
        assert snapshot_started.wait(5)
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(recorder.record, event("B-before")).result(5)
        for name in names[:2]:
            thread = threading.Thread(
                target=actor, args=(name,), name=name, daemon=True
            )
            threads.append(thread)
            thread.start()
            assert waiting[name].wait(5)
        # A later event is exported before the B event covered by both flushes.
        assert recorder.record(event("A-after"))
        thread = threading.Thread(
            target=actor, args=("close",), name="close", daemon=True
        )
        threads.append(thread)
        thread.start()
        assert waiting["close"].wait(5)
        release_snapshot.set()
        assert b_write_started.wait(5)
        with recorder._condition:
            assert not any(decisions[name][-1] for name in names)
            probe_barriers.set()
            recorder._condition.notify_all()
        for name in names:
            assert rechecked[name].wait(5)
        assert [r["operation_id"] for r in read_records(output)] == [
            "A-before", "A-after",
        ]
        assert recorder.status.valid
        assert {name: decisions[name][-1] for name in names} == {
            "flush-one": False, "flush-two": False, "close": False,
        }
        assert not any(done.is_set() for done in finished.values())
    finally:
        release_snapshot.set()
        release_b_write.set()
        for thread in threads:
            thread.join(5)
        if not any(thread.is_alive() for thread in threads):
            recorder.close()
    assert not any(thread.is_alive() for thread in threads)
    assert all(done.is_set() for done in finished.values())
    assert [r["operation_id"] for r in read_records(output)] == [
        "A-before", "A-after", "B-before",
    ]
    assert recorder.status.valid


@pytest.mark.parametrize("failure", ["error", "short_write"])
def test_concurrent_flush_and_close_wake_on_export_failure(
    tmp_path, monkeypatch, failure
):
    write_started = threading.Event()
    release_write = threading.Event()
    waiting = {name: threading.Event() for name in ("f1", "f2", "close")}
    real_write = os.write

    def write(fd, data):
        write_started.set()
        assert release_write.wait(5)
        if failure == "error":
            raise OSError("export unavailable")
        return real_write(fd, data[:5])

    monkeypatch.setattr(os, "write", write)
    recorder = ShardedRingRecorder(capacity=1, output=tmp_path / "e.jsonl")
    real_wait_for = recorder._condition.wait_for

    def wait_for(predicate, timeout=None):
        name = threading.current_thread().name
        if name in waiting:
            waiting[name].set()
        return real_wait_for(predicate, timeout)

    monkeypatch.setattr(recorder._condition, "wait_for", wait_for)
    finished = {name: threading.Event() for name in waiting}

    def actor(name):
        (recorder.close if name == "close" else recorder.flush)()
        finished[name].set()

    threads = []
    try:
        assert recorder.record(event("one"))
        assert write_started.wait(5)
        for name in waiting:
            thread = threading.Thread(
                target=actor, args=(name,), name=name, daemon=True
            )
            threads.append(thread)
            thread.start()
            assert waiting[name].wait(5)
        assert not any(done.is_set() for done in finished.values())
    finally:
        release_write.set()
        for thread in threads:
            thread.join(5)
        if not any(thread.is_alive() for thread in threads):
            recorder.close()
    assert all(done.is_set() for done in finished.values())
    assert not any(thread.is_alive() for thread in threads)
    assert not recorder.status.valid
    assert "export_error" in recorder.status.reasons
    assert not recorder.record(lambda: pytest.fail("closed factory evaluated"))


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity_rejects_nonpositive_or_noninteger_values(tmp_path, capacity):
    with pytest.raises(ValueError):
        ShardedRingRecorder(capacity=capacity, output=tmp_path / "e.jsonl")


def test_validation_accepts_out_of_order_dependencies_and_missing_times():
    records = [
        event("child", parent_operation_id="root", dependency_ids=("root",),
              enqueue_ns=1, start_ns=2, end_ns=3).to_record(),
        event("root").to_record(),
    ]
    report = validate_trace(records, TraceRunStatus())
    assert report.valid
    assert not report.reasons


@pytest.mark.parametrize("change,reason", [
    ({"schema_version": 2}, "schema_version"),
    ({"operation_id": ""}, "operation_id"),
    ({"start_ns": 3, "end_ns": 2}, "span_order"),
    ({"enqueue_ns": 3, "start_ns": 2}, "span_order"),
    ({"enqueue_ns": 3, "end_ns": 2}, "span_order"),
    ({"device_start": 3.0, "device_end": 2.0}, "span_order"),
    ({"start_ns": "bad"}, "span_time"),
    ({"device_start": float("nan")}, "span_time"),
    ({"dependency_ids": ["missing"]}, "missing_dependency"),
    ({"parent_operation_id": "missing"}, "missing_dependency"),
    ({"dependency_ids": ["one"]}, "self_dependency"),
    ({"dependency_ids": "one"}, "dependency_ids"),
])
def test_validation_rejects_structural_errors_without_raising(change, reason):
    record = event("one").to_record()
    record.update(change)
    report = validate_trace([record], TraceRunStatus())
    assert not report.valid
    assert reason in report.reasons


def test_validation_rejects_duplicate_operations_and_invalid_status():
    record = event("one").to_record()
    assert "duplicate_operation_id" in validate_trace(
        [record, record], TraceRunStatus()
    ).reasons
    report = validate_trace([], TraceRunStatus(dropped_events=1))
    assert not report.valid
    assert "dropped_event" in report.reasons
    report = validate_trace([], TraceRunStatus(valid=False,
                                               reasons=("export_error",)))
    assert not report.valid
    assert "export_error" in report.reasons


def test_validation_malformed_records_are_invalid_not_serving_exceptions():
    report = validate_trace([None, {"operation_id": []}], TraceRunStatus())
    assert not report.valid
    assert "record_type" in report.reasons
