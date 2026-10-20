import asyncio
import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from aibrix_kvcache.compression_profiler.operation import OperationTrace
from aibrix_kvcache.integration.vllm.hybrid.layout import HybridLayout
from aibrix_kvcache.integration.vllm.hybrid.store import (
    HybridStore,
    HybridStoreError,
)
from aibrix_kvcache.l2.connectors.connector import Connector, ConnectorConfig

NUM_BLOCKS = 6
GROUPS = {0: {"linear": "mamba"}, 4: {"ring": "skip"}, 5: {"attn": "attention"}}


def _caches():
    torch.manual_seed(0)
    return {
        "linear": (
            torch.randn(NUM_BLOCKS, 3, 5),
            torch.randn(NUM_BLOCKS, 2).to(torch.bfloat16),
        ),
        "attn": torch.randn(2, NUM_BLOCKS, 4).to(torch.bfloat16),
        "ring": torch.zeros(NUM_BLOCKS, 2),
    }


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_MEMORY_CAPACITY_BYTES", "1000000")
    connector = Connector.create(
        ConnectorConfig(
            backend_name="MEMORY",
            namespace="hybrid-test",
            partition_id="p0",
            executor=ThreadPoolExecutor(1),
        )
    )
    assert connector.open().is_ok()
    yield connector
    connector.close()


def _store(backend, caches, codec, chunk_bytes=16, **kwargs):
    layout = HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS)
    # Tiny chunks force several staging copies even for test-sized records.
    return HybridStore(
        layout, backend, codec, chunk_bytes=chunk_bytes, **kwargs
    )


def test_save_evict_load_restores_exact_bytes(backend):
    caches = _caches()
    store = _store(backend, caches, "IDENTITY")
    source = {0: [2], 5: [1, 3]}
    original = store.layout.gather(source)

    stats = store.save(b"key-1", source)
    for tensor in (*caches["linear"], caches["attn"]):
        tensor.zero_()
    target = {0: [4], 5: [0, 5]}
    load_stats = store.load(b"key-1", target)

    assert store.layout.gather(target) == original
    assert stats["raw_bytes"] == sum(map(len, original))
    assert stats["stored_bytes"] > 0
    assert load_stats["stored_bytes"] == stats["stored_bytes"]


def test_cpu_restore_does_not_enter_a_cuda_stream(backend, monkeypatch):
    store = _store(backend, _caches(), "IDENTITY")
    source, target = {0: [2], 5: [1, 3]}, {0: [4], 5: [0, 5]}
    original = store.layout.gather(source)
    store.save(b"cpu-key", source)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU restore entered a CUDA stream")

    monkeypatch.setattr(torch.cuda, "stream", forbidden)
    try:
        store.load(b"cpu-key", target)
        assert store.layout.gather(target) == original
    finally:
        store.close()


def test_state_and_spans_restore_a_record_part_by_part(backend):
    caches = _caches()
    store = _store(backend, caches, "IDENTITY")
    parts = {
        b"state": ({0: [2], 5: []}, {0: [4], 5: []}),
        b"span-0": ({0: [], 5: [1]}, {0: [], 5: [0]}),
        b"span-1": ({0: [], 5: [3]}, {0: [], 5: [5]}),
    }
    original = store.layout.gather({0: [2], 5: [1, 3]})
    for key, (source, _) in parts.items():
        store.save(key, source)
    for tensor in (*caches["linear"], caches["attn"]):
        tensor.zero_()
    for key, (_, target) in parts.items():
        store.load(key, target)
    assert store.layout.gather({0: [4], 5: [0, 5]}) == original


def test_a_stored_key_is_not_stored_again(backend):
    # E.g. after a save failed on another rank: the re-save must not fail
    # here because recomputed blocks differ in their last bits.
    caches = _caches()
    store = _store(backend, caches, "IDENTITY")
    source = {0: [2], 5: [1, 3]}
    original = store.layout.gather(source)
    first = store.save(b"key-1", source)
    caches["attn"].add_(1)
    again = store.save(b"key-1", source)
    target = {0: [4], 5: [0, 5]}
    store.load(b"key-1", target)
    assert again == {**first, "stored_bytes": 0}
    assert store.layout.gather(target) == original
    store.close()


def test_load_missing_key_raises_without_touching_cache(backend):
    caches = _caches()
    store = _store(backend, caches, "IDENTITY")
    before = caches["attn"].clone()
    with pytest.raises(HybridStoreError, match="not found"):
        store.load(b"absent", {0: [1], 5: [0, 1]})
    assert torch.equal(caches["attn"], before)


def test_load_rejects_record_from_another_layout(backend):
    caches = _caches()
    _store(backend, caches, "IDENTITY").save(b"key", {0: [2], 5: [1, 3]})
    other = _caches()
    other["attn"] = other["attn"].float()
    store = _store(backend, other, "IDENTITY")
    before = other["attn"].clone()
    with pytest.raises(HybridStoreError, match="integrity"):
        store.load(b"key", {0: [4], 5: [0, 5]})
    assert torch.equal(other["attn"], before)


def test_corrupted_record_is_rejected(backend):
    caches = _caches()
    store = _store(backend, caches, "IDENTITY")
    store.save(b"key", {0: [2], 5: [1, 3]})
    payload = bytearray(asyncio.run(backend.get_bytes(b"key")).value)
    payload[-3] ^= 0xFF
    before = caches["attn"].clone()
    assert asyncio.run(backend.delete(b"key")).is_ok()
    assert asyncio.run(backend.put_bytes(b"key", bytes(payload))).is_ok()
    with pytest.raises(HybridStoreError, match="integrity"):
        store.load(b"key", {0: [4], 5: [0, 5]})
    assert torch.equal(caches["attn"], before)


def test_compare_reports_which_slots_differ_without_writing(backend):
    caches = _caches()
    store = _store(backend, caches, "IDENTITY")
    store.save(b"key", {0: [2], 5: [1, 3]})
    caches["attn"][:, 3] += 1
    before = caches["attn"].clone()

    report = store.compare(b"key", {0: [2], 5: [1, 3]})

    assert [(r["name"], r["equal"]) for r in report] == [
        ("linear", True),
        ("linear", True),
        ("attn", False),
    ]
    assert torch.equal(caches["attn"], before)


def test_load_readback_confirms_target_blocks_hold_record(backend):
    caches = _caches()
    store = _store(backend, caches, "IDENTITY")
    store.save(b"key", {0: [2], 5: [1, 3]})
    stats = store.load(b"key", {0: [4], 5: [0, 5]}, readback=True)
    assert stats["readback_equal"] is True
    plain = store.load(b"key", {0: [4], 5: [0, 5]})
    assert "readback_equal" not in plain


ON_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("nvidia.nvcomp") is None,
    reason="needs CUDA and nvidia-nvcomp",
)


@pytest.mark.parametrize(
    "codec", ["IDENTITY", pytest.param("ANS", marks=ON_GPU)]
)
def test_save_evict_load_is_one_valid_profiler_resume_chain(
    backend, tmp_path, codec
):
    """Cold save, then another request's local miss and external restore."""
    from aibrix_kvcache.compression_profiler import (
        EventType,
        ProfileMode,
        TraceContext,
    )
    from aibrix_kvcache.compression_profiler.operation import (
        OperationTrace,
        TraceSink,
        request_event,
    )
    from aibrix_kvcache.compression_profiler.recorder import (
        ShardedRingRecorder,
    )
    from aibrix_kvcache.compression_profiler.validation import (
        validate_qwen_resume_trace,
    )

    output = tmp_path / "trace.jsonl"
    recorder = ShardedRingRecorder(capacity=64, output=output)
    caches = _caches()
    if codec == "ANS":
        caches = {
            name: tuple(t.cuda() for t in value)
            if isinstance(value, tuple)
            else value.cuda()
            for name, value in caches.items()
        }
    store = _store(backend, caches, codec)
    sink = TraceSink(
        recorder, ProfileMode.LIGHT, frozenset(), codec, "0,5", "sig"
    )

    contexts = {
        request: TraceContext("exp", "run", request, ProfileMode.LIGHT, request)
        for request in ("cold", "resumed")
    }
    for context in contexts.values():
        request_event(recorder, context, OperationTrace.now())

    def trace(request):
        return OperationTrace(sink, contexts[request])

    saved = store.save(b"key", {0: [2], 5: [1, 3]}, trace("cold"))
    evict = trace("resumed")
    start = evict.clock()
    evicted = evict.event(
        EventType.EVICTION, b"key", start, (saved["put"],),
        result="local_miss",
    )
    evict.finish()
    store.load(
        b"key", {0: [4], 5: [0, 5]}, trace=trace("resumed"),
        after=(evicted,),
    )
    recorder.flush()
    records = [json.loads(line) for line in output.read_text().splitlines()]

    report = validate_qwen_resume_trace(records, recorder.status)
    assert report.valid, report
    assert not report.unknowns, report
    recorder.close()
    store.close()


def test_save_async_snapshots_blocks_before_returning(backend):
    caches = _caches()
    store = _store(backend, caches, "IDENTITY")
    source = {0: [2], 5: [1, 3]}
    original = store.layout.gather_flat(source).clone()

    future = store.save_async(b"key", source)
    # The GPU blocks may be reused as soon as save_async returns.
    for tensor in (*caches["linear"], caches["attn"]):
        tensor.zero_()
    stats = future.result(timeout=10)

    assert stats["raw_bytes"] == original.numel()
    store.load(b"key", {0: [4], 5: [0, 5]})
    assert torch.equal(store.layout.gather_flat({0: [4], 5: [0, 5]}), original)
    store.close()


def test_save_async_reports_backend_failure(monkeypatch):
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_MEMORY_CAPACITY_BYTES", "8")
    tiny = Connector.create(
        ConnectorConfig(
            backend_name="MEMORY",
            namespace="tiny",
            partition_id="p0",
            executor=ThreadPoolExecutor(1),
        )
    )
    assert tiny.open().is_ok()
    store = _store(tiny, _caches(), "IDENTITY")
    future = store.save_async(b"key", {0: [2], 5: [1, 3]})
    with pytest.raises(HybridStoreError, match="put failed"):
        future.result(timeout=10)
    store.close()


def test_failed_staging_is_a_store_error_and_frees_its_buffer(
    backend, monkeypatch
):
    """The connector skips a save that raises HybridStoreError; anything
    else (e.g. out of device memory) would kill the engine."""
    store = _store(backend, _caches(), "IDENTITY", max_inflight=1)

    def out_of_memory(*_):
        raise RuntimeError("CUDA Runtime failure: out of memory")

    monkeypatch.setattr(store.layout, "gather_into", out_of_memory)
    for _ in range(2):  # the second would block on a leaked buffer
        with pytest.raises(HybridStoreError, match="out of memory"):
            store.save_async(b"key", {0: [2], 5: [1, 3]})
    store.close()


def test_a_save_is_dropped_not_waited_for_while_earlier_ones_are_busy(
    backend,
):
    """A stuck save must cost its own record, never stall the worker
    (vLLM declares the engine dead when a step takes too long)."""
    store = _store(backend, _caches(), "IDENTITY", max_inflight=2)
    release = threading.Event()
    store._saver.submit(release.wait)  # a save stuck in the background
    busy = [store.save_async(f"k{i}".encode(), {0: [2], 5: [1]})
            for i in range(2)]
    begin = time.perf_counter()
    with pytest.raises(HybridStoreError, match="dropped"):
        store.save_async(b"k2", {0: [2], 5: [1]})
    assert time.perf_counter() - begin < 1
    release.set()
    for future in busy:
        future.result(timeout=10)
    store.save(b"k2", {0: [2], 5: [1]})  # buffers come back
    assert store.staging_buffers == 2
    store.close()


def test_background_encode_releases_the_gil(backend):
    """The worker thread must keep running while a record is encoded."""
    big = {
        "linear": (
            torch.randn(NUM_BLOCKS, 4096, 1024),
            torch.randn(NUM_BLOCKS, 2).to(torch.bfloat16),
        ),
        "attn": torch.randn(2, NUM_BLOCKS, 4).to(torch.bfloat16),
        "ring": torch.zeros(NUM_BLOCKS, 2),
    }
    store = _store(backend, big, "IDENTITY", chunk_bytes=1 << 20)
    backend.capacity_bytes = 1 << 30

    def rate(future):
        """Iterations per second of a worker-like loop until ``future``."""
        ticks, begin = 0, time.perf_counter()
        while not future.done():
            ticks += 1
        return ticks / (time.perf_counter() - begin)

    alone = rate(store._saver.submit(time.sleep, 0.05))
    during = rate(store.save_async(b"key", {0: [0, 1, 2, 3, 4, 5], 5: [1]}))
    # A GIL-holding save leaves the loop a few slices per switch interval;
    # the machine's speed cancels out of the ratio.
    assert during > alone / 4, (during, alone)
    store.close()


@pytest.fixture
def light_trace(tmp_path):
    """(new trace, records reader) writing one LIGHT request's events."""
    from aibrix_kvcache.compression_profiler import ProfileMode, TraceContext
    from aibrix_kvcache.compression_profiler.operation import (
        OperationTrace,
        TraceSink,
    )
    from aibrix_kvcache.compression_profiler.recorder import (
        ShardedRingRecorder,
    )

    output = tmp_path / "trace.jsonl"
    recorder = ShardedRingRecorder(capacity=64, output=output)
    sink = TraceSink(
        recorder, ProfileMode.LIGHT, frozenset(), "IDENTITY", "0,5", "sig"
    )
    context = TraceContext("exp", "run", "r", ProfileMode.LIGHT, "r")

    def records():
        recorder.flush()
        return [json.loads(line) for line in output.read_text().splitlines()]

    yield lambda: OperationTrace(sink, context), records
    recorder.close()


def test_export_separates_saver_queue_from_its_device_window(
    backend, light_trace
):
    new_trace, records = light_trace
    store = _store(backend, _caches(), "IDENTITY")
    release, released = threading.Event(), []
    # An earlier save still encoding holds the single saver thread.
    store._saver.submit(release.wait)

    future = store.save_async(b"key", {0: [2], 5: [1, 3]}, new_trace())
    released.append(OperationTrace.now())
    release.set()
    future.result()

    (export,) = [r for r in records() if r["event_type"] == "kv_export"]
    # Waiting behind that save is queue time, never the export's own window.
    assert export["enqueue_ns"] < released[0] <= export["start_ns"]
    store.close()


def test_verify_import_window_covers_the_comparison(
    backend, light_trace, monkeypatch
):
    new_trace, records = light_trace
    store = _store(backend, _caches(), "IDENTITY")
    store.save(b"key", {0: [2], 5: [1, 3]})
    gather, compared = store.layout.gather, []

    def timed_gather(block_ids):
        compared.append(OperationTrace.now())
        return gather(block_ids)

    monkeypatch.setattr(store.layout, "gather", timed_gather)
    store.compare(b"key", {0: [2], 5: [1, 3]}, new_trace())

    (verify,) = [r for r in records() if r["event_type"] == "kv_import"]
    assert verify["phase"] == "verify"
    # The comparison it reports happens inside its window.
    assert verify["start_ns"] <= compared[0] <= verify["end_ns"]
    store.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_traced_transfers_carry_their_device_time(backend, light_trace):
    new_trace, records = light_trace
    caches = {
        name: tuple(t.cuda() for t in value) if isinstance(value, tuple)
        else value.cuda()
        for name, value in _caches().items()
    }
    store = _store(backend, caches, "IDENTITY")
    store.save(b"key", {0: [2], 5: [1, 3]}, new_trace())
    store.load(b"key", {0: [4], 5: [0, 5]}, trace=new_trace())

    transfers = [
        r for r in records() if r["event_type"] in ("kv_export", "kv_import")
    ]
    assert len(transfers) == 2
    for row in transfers:
        assert row["device_start"] == 0
        # The copies may finish while the save still queues for the saver.
        begin = row.get("enqueue_ns") or row["start_ns"]
        assert 0 < row["device_end"] <= row["end_ns"] - begin
    store.close()


def test_untraced_or_host_transfers_have_no_device_time(backend, light_trace):
    new_trace, records = light_trace
    store = _store(backend, _caches(), "IDENTITY")
    store.save(b"key", {0: [2], 5: [1, 3]}, new_trace())
    (export,) = [r for r in records() if r["event_type"] == "kv_export"]
    assert export["device_start"] is None and export["device_end"] is None
    store.close()


def test_restore_window_ends_before_its_diagnostic_readback(
    backend, light_trace, monkeypatch
):
    new_trace, records = light_trace
    store = _store(backend, _caches(), "IDENTITY")
    store.save(b"key", {0: [2], 5: [1, 3]})
    gather_flat, readback = store.layout.gather_flat, []

    def timed_gather_flat(block_ids):
        readback.append(OperationTrace.now())
        return gather_flat(block_ids)

    monkeypatch.setattr(store.layout, "gather_flat", timed_gather_flat)
    store.load(b"key", {0: [4], 5: [0, 5]}, readback=True, trace=new_trace())

    (restore,) = [r for r in records() if r["event_type"] == "kv_import"]
    assert restore["restore_result"] == "readback_equal"
    assert restore["end_ns"] <= readback[0]
    store.close()
