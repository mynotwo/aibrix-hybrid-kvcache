"""Regressions for context snapshots and completed-read DAGs."""

from concurrent.futures import ThreadPoolExecutor
from graphlib import TopologicalSorter

import pytest
import torch
import torch.distributed as dist

from aibrix_kvcache import GroupAwareKVCacheManager
from aibrix_kvcache.compression_profiler import EventType
from aibrix_kvcache.status import Status, StatusCodes

from .test_compression_profiler_context import (
    cache_factory as cache_factory,
)
from .test_compression_profiler_context import (
    context,
    records,
    worker_fixture,
)


def seed(cache, tokens):
    handle = cache.allocate_for(None, tokens).get()
    assert cache.put(None, tokens, handle).is_ok()


def receive(worker, meta, kind):
    if kind == 1:
        return worker._recv_kv_sync_impl(meta)
    received = worker._recv_kv_impl(meta)
    metadata = {meta.req_id: meta}
    worker._start_load_kv(metadata, "layer.4")
    worker.wait_for_layer_load(metadata, "layer.4")
    worker.wait_for_layer_load(metadata, "layer.9")
    return received


@pytest.mark.parametrize("kind", [1, 2])
def test_acquire_completion_cannot_rebind_import_attempt(
    cache_factory, monkeypatch, kind
):
    cache, tokens, output = cache_factory()
    seed(cache, tokens)
    worker, meta, _ = worker_fixture(kind, cache, tokens, monkeypatch)
    acquire = cache.acquire

    def acquire_and_update(*args, **kwargs):
        result = acquire(*args, **kwargs)
        meta.trace_context = meta.trace_context.child("updated")
        return result

    monkeypatch.setattr(cache, "acquire", acquire_and_update)
    assert receive(worker, meta, kind) == len(tokens)
    events = records(cache, output)
    imports = [e for e in events if e["event_type"] == "kv_import"]
    acquired = [e for e in events if e["phase"] == "cache_acquire"]
    assert imports and acquired
    assert {e["parent_operation_id"] for e in imports + acquired} == {"initial"}


@pytest.mark.parametrize("direction", ["import", "export"])
def test_type2_launch_failure_is_not_later_reported_as_success(
    cache_factory, monkeypatch, direction
):
    cache, tokens, output = cache_factory()
    seed(cache, tokens)
    worker, meta, ns = worker_fixture(2, cache, tokens, monkeypatch)
    metadata = {meta.req_id: meta}
    failure = RuntimeError("kernel launch rejected")
    if direction == "import":
        assert worker._recv_kv_impl(meta) == len(tokens)
        ns["reshape_and_cache_multi_layer"].side_effect = failure
        launch = lambda: worker._start_load_kv(metadata, "layer.9")
        wait = lambda: worker.wait_for_layer_load(metadata, "layer.9")
    else:
        ns["reshape_and_offload_multi_layer"].side_effect = failure
        launch = lambda: worker.save_kv_layer(
            metadata, "layer.4", torch.zeros(1), None
        )
        wait = lambda: worker.wait_for_save(metadata)
    with pytest.raises(RuntimeError) as raised:
        launch()
    assert raised.value is failure
    wait()
    events = [
        e
        for e in records(cache, output)
        if e["event_type"] == "kv_" + direction
    ]
    assert len(events) == 1
    assert events[0]["result"] == "error"
    assert events[0]["error_class"] == "RuntimeError"
    if direction == "import":
        assert events[0]["restore_result"] == "rejected"
    assert worker._import_spans == worker._export_spans == []


def ancestors(events, event):
    by_id = {e["operation_id"]: e for e in events}
    pending = list(event["dependency_ids"])
    pending.append(event["parent_operation_id"])
    found = set()
    while pending:
        operation = pending.pop()
        if operation in found or operation not in by_id:
            continue
        found.add(operation)
        prior = by_id[operation]
        pending.extend(prior["dependency_ids"])
        pending.append(prior["parent_operation_id"])
    return found


def assert_closed_acyclic(events):
    graph = {
        e["operation_id"]: e["dependency_ids"] + (
            [e["parent_operation_id"]] if e["parent_operation_id"] else []
        )
        for e in events
    }
    assert all(set(edges) <= graph.keys() for edges in graph.values())
    assert set(TopologicalSorter(graph).static_order()) == graph.keys()


@pytest.fixture(params=[False, True], ids=["base", "group"])
def acquire_manager(request, tmp_path):
    if not request.param:
        yield {}
        return
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "group").as_uri(), rank=0, world_size=1
    )
    try:
        yield {
            "manager": GroupAwareKVCacheManager,
            "process_group": dist.group.WORLD,
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("mode", ["LIGHT", "DETAILED"])
@pytest.mark.parametrize("codec", ["", "IDENTITY"])
def test_import_reaches_its_completed_cache_and_l2_read(
    cache_factory, monkeypatch, tmp_path, acquire_manager, kind, mode, codec
):
    if codec:
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND", "SHFS")
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_SHFS_ROOT", str(tmp_path / "l2"))
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_L2_CACHE_COMPRESSION", codec)
    cache, tokens, output = cache_factory(mode=mode, **acquire_manager)
    seed(cache, tokens)
    worker, meta, _ = worker_fixture(kind, cache, tokens, monkeypatch)
    cache.trace_finish((meta.trace_context, None), EventType.REQUEST)
    assert receive(worker, meta, kind) == len(tokens)
    events = records(cache, output)
    reads = [e for e in events if e["event_type"] == "l2_get"]
    acquired = [e for e in events if e["phase"] == "cache_acquire"]
    imports = [e for e in events if e["phase"].startswith("connector_")]
    assert reads and acquired and imports
    assert_closed_acyclic(events)
    for completed in acquired:
        assert {e["operation_id"] for e in reads} <= ancestors(
            events, completed
        )
    for imported in imports:
        assert imported["dependency_ids"] == [acquired[0]["operation_id"]]
        reachable = ancestors(events, imported)
        assert {e["operation_id"] for e in reads + acquired} <= reachable
        if codec:
            assert {
                e["operation_id"]
                for e in events
                if e["event_type"] == "codec_decode" or e["phase"] == "restore"
            } <= reachable


def test_shared_mr_consumers_keep_separate_dependency_snapshots(cache_factory):
    cache, tokens, output = cache_factory(l1=True)
    seed(cache, tokens)

    def acquire(request):
        return cache.acquire(
            None, tokens, trace_context=context(request)
        ).get()[1]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(acquire, ("req-a", "req-b"))
    try:
        assert first.memory_regions[0] is second.memory_regions[0]
        first_ids, second_ids = (
            first._profiler_dependencies,
            second._profiler_dependencies,
        )
        assert isinstance(first_ids, tuple) and isinstance(second_ids, tuple)
        assert (
            first_ids and second_ids and set(first_ids).isdisjoint(second_ids)
        )
        events = {e["operation_id"]: e for e in records(cache, output)}
        assert {events[i]["request_id"] for i in first_ids} == {"req-a"}
        assert {events[i]["request_id"] for i in second_ids} == {"req-b"}
    finally:
        first.release()
        second.release()


@pytest.mark.parametrize("mode", ["LIGHT", "DETAILED"])
@pytest.mark.parametrize("rejected", ["l2_get", "cache_acquire", "both"])
@pytest.mark.parametrize("codec", ["", "IDENTITY"])
def test_rejected_records_never_become_import_dependencies(
    cache_factory, monkeypatch, tmp_path, mode, rejected, codec
):
    if codec:
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND", "SHFS")
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_SHFS_ROOT", str(tmp_path / "l2"))
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_L2_CACHE_COMPRESSION", codec)
    cache, tokens, output = cache_factory(mode=mode)
    seed(cache, tokens)
    worker, meta, _ = worker_fixture(1, cache, tokens, monkeypatch)
    cache.trace_finish((meta.trace_context, None), EventType.REQUEST)
    recorder = cache._l2_cache._profiler_recorder
    record = recorder.record

    def reject_read(factory):
        event = factory()
        reject_get = rejected in ("l2_get", "both")
        reject_acquire = rejected in ("cache_acquire", "both")
        if (reject_get and event.event_type.value == "l2_get") or (
            reject_acquire and event.phase == "cache_acquire"
        ):
            return False
        return record(lambda: event)

    monkeypatch.setattr(recorder, "record", reject_read)
    assert receive(worker, meta, 1) == len(tokens)
    events = records(cache, output)
    assert_closed_acyclic(events)
    assert not recorder.status.valid
    assert "observation_error" in recorder.status.reasons
    if codec and rejected in ("l2_get", "both"):
        assert not any(
            e["event_type"] == "codec_decode" or e["phase"] == "restore"
            for e in events
        )
    imported = next(e for e in events if e["phase"] == "connector_sync")
    if rejected in ("cache_acquire", "both"):
        assert imported["dependency_ids"] == []
    else:
        acquired = next(e for e in events if e["phase"] == "cache_acquire")
        assert imported["dependency_ids"] == [acquired["operation_id"]]


def test_rejected_acquire_miss_does_not_leave_dangling_fallback(
    cache_factory, monkeypatch
):
    cache, tokens, output = cache_factory()
    root = context().child("initial")
    cache.trace_finish((root, None), EventType.REQUEST)
    recorder = cache._l2_cache._profiler_recorder
    record = recorder.record

    def reject_acquire(factory):
        event = factory()
        if (
            event.event_type == EventType.REQUEST
            and event.phase == "cache_acquire"
        ):
            return False
        return record(lambda: event)

    monkeypatch.setattr(recorder, "record", reject_acquire)
    assert cache.acquire(None, tokens, trace_context=root).is_not_found()
    assert_closed_acyclic(records(cache, output))
    assert not recorder.status.valid


@pytest.mark.parametrize("reject", [False, True])
def test_light_error_depends_on_accepted_aggregate(
    cache_factory, monkeypatch, reject
):
    cache, tokens, output = cache_factory(mode="LIGHT")
    root = context().child("initial")
    cache.trace_finish((root, None), EventType.REQUEST)
    recorder = cache._l2_cache._profiler_recorder
    record = recorder.record

    async def failed_get(*args):
        return Status(StatusCodes.ERROR, "backend failure")

    def reject_get(factory):
        event = factory()
        if reject and event.event_type == EventType.L2_GET:
            return False
        return record(lambda: event)

    monkeypatch.setattr(cache._l2_cache._backend, "get", failed_get)
    monkeypatch.setattr(recorder, "record", reject_get)
    result = cache.acquire(None, tokens, trace_context=root)
    assert result.error_code == StatusCodes.ERROR
    events = records(cache, output)
    assert_closed_acyclic(events)
    errors = [e for e in events if e["event_type"] == "error_recovery"]
    assert len(errors) == (0 if reject else 1)
    assert recorder.status.valid is not reject


def test_off_does_not_add_result_sideband_or_handle_attributes(
    cache_factory, monkeypatch
):
    cache, tokens, output = cache_factory(mode="OFF")
    seed(cache, tokens)
    get = cache._l2_cache.get

    async def observe(*args, **kwargs):
        assert kwargs.get("trace_result") is None
        return await get(*args, **kwargs)

    monkeypatch.setattr(cache._l2_cache, "get", observe)
    handle = cache.acquire(None, tokens, trace_context=context()).get()[1]
    try:
        assert not hasattr(handle, "_profiler_dependencies")
        assert not output.exists()
    finally:
        handle.release()


@pytest.mark.parametrize("direction", ["import", "export"])
def test_observation_failure_cannot_replace_launch_exception(
    cache_factory, monkeypatch, direction
):
    cache, tokens, _ = cache_factory()
    seed(cache, tokens)
    worker, meta, ns = worker_fixture(2, cache, tokens, monkeypatch)
    metadata = {meta.req_id: meta}
    failure = RuntimeError("kernel rejected")

    def record_failure(factory):
        raise ValueError("recorder rejected")

    recorder = cache._l2_cache._profiler_recorder
    monkeypatch.setattr(recorder, "record", record_failure)
    with pytest.raises(RuntimeError) as raised:
        if direction == "import":
            assert worker._recv_kv_impl(meta) == len(tokens)
            ns["reshape_and_cache_multi_layer"].side_effect = failure
            worker._start_load_kv(metadata, "layer.9")
        else:
            ns["reshape_and_offload_multi_layer"].side_effect = failure
            worker.save_kv_layer(metadata, "layer.4", torch.zeros(1), None)
    assert raised.value is failure
    assert worker._import_spans == worker._export_spans == []
    assert "observation_error" in recorder.status.reasons
    worker._release_acquired_kvcache_handles()
    worker._release_allocated_kvcache_handles()
