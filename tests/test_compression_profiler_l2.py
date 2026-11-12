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

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from aibrix_kvcache.cache_hashable import TokenListView
from aibrix_kvcache.compression_profiler import (
    ProfileMode,
    TraceContext,
    TraceEvent,
)
from aibrix_kvcache.compression_profiler.recorder import NullRecorder
from aibrix_kvcache.compression_profiler.validation import validate_trace
from aibrix_kvcache.l2 import KeyBuilder, L2Cache
from aibrix_kvcache.memory import TensorPoolAllocator
from aibrix_kvcache.status import Status, StatusCodes

from .conftest import release_mrs
from .test_l2cache import build_get_mrs, build_put_mrs

ENV = "AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_"


@pytest.fixture
def setup_cache(cache_conf_fixture, tmp_path, monkeypatch):
    _, spec = cache_conf_fixture
    root = tmp_path / "shfs"
    output = tmp_path / "events.jsonl"
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_SHFS_ROOT", str(root))
    monkeypatch.setenv(ENV + "OUTPUT", str(output))
    monkeypatch.setenv(ENV + "REQUEST_IDS", "target")
    for setting in ("MODE", "CAPACITY"):
        monkeypatch.delenv(ENV + setting, raising=False)
    caches = []
    regions = []
    with ThreadPoolExecutor(max_workers=2) as executor:

        def create(mode="DETAILED", codec="ZSTD", backend="SHFS"):
            if mode is not None:
                monkeypatch.setenv(ENV + "MODE", mode)
            cache = L2Cache(
                backend_name=backend,
                placement_policy="SIMPLE",
                namespace="profiler",
                block_spec=spec,
                executor=executor,
                compression=codec,
                key_builder=KeyBuilder.create(
                    "ROLLING_HASH", block_size=spec.block_ntokens
                ),
            )
            caches.append(cache)
            cache.open().raise_if_not_ok()
            allocator = TensorPoolAllocator.create(
                capacity_nbytes=16 * spec.block_nbytes
            )
            tokens = TokenListView(list(range(2 * spec.block_ntokens)))
            args = (
                allocator,
                spec.block_nbytes,
                spec.block_ntokens,
                None,
                tokens,
            )
            source, destination = build_put_mrs(*args), build_get_mrs(*args)
            regions.extend(source + destination)
            return cache, tokens, source, destination

        yield create, root, output
        for cache in caches:
            cache.close()
        release_mrs(regions)


def context(request="target", mode=ProfileMode.DETAILED):
    return TraceContext("experiment", "run", request, mode)


def records(cache, output):
    cache._profiler_recorder.flush()
    return [json.loads(line) for line in output.read_text().splitlines()]


@pytest.mark.asyncio
@pytest.mark.parametrize("codec", ["ZSTD", "IDENTITY"])
async def test_detailed_dependency_bytes_restore(setup_cache, codec):
    create, root, output = setup_cache
    cache, tokens, source, destination = create(codec=codec)
    assert (
        await cache.put(None, tokens, source, trace_context=context())
    ).is_ok()
    events = records(cache, output)
    assert len(events) == 4
    for key in {event["block_key"] for event in events}:
        encode, store = [e for e in events if e["block_key"] == key]
        assert [encode["event_type"], store["event_type"]] == [
            "codec_encode",
            "l2_put",
        ]
        assert store["dependency_ids"] == [encode["operation_id"]]
        assert (
            encode["payload_bytes"]
            + encode["metadata_bytes"]
            + encode["framing_bytes"]
        ) == encode["envelope_bytes"]
        # Persisted metadata stores the hex key, independent of filename layout.
        matching = [
            p
            for p in root.rglob("*")
            if p.is_file() and key.encode() in p.read_bytes()
        ]
        assert len(matching) == 1
        assert store["connector_write_bytes"] == matching[0].stat().st_size
    assert (
        await cache.get(None, tokens, destination, trace_context=context())
    ).is_ok()
    events = records(cache, output)[4:]
    for key in {event["block_key"] for event in events}:
        load, decode, restore = [e for e in events if e["block_key"] == key]
        assert [e["event_type"] for e in (load, decode, restore)] == [
            "l2_get",
            "codec_decode",
            "kv_import",
        ]
        assert decode["dependency_ids"] == [load["operation_id"]]
        assert restore["dependency_ids"] == [decode["operation_id"]]
        assert restore["phase"] == "restore"
        assert restore["restore_result"] == "restored"
        assert all(
            e["start_ns"] <= e["end_ns"] for e in (load, decode, restore)
        )
    assert [mr.tobytes() for mr in destination] == [
        mr.tobytes() for mr in source
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,request_id", [("LIGHT", "target"), ("DETAILED", "other")]
)
async def test_light_aggregates_blocks(setup_cache, mode, request_id):
    create, root, output = setup_cache
    cache, tokens, source, destination = create(mode)
    assert (
        await cache.put(None, tokens, source, trace_context=context(request_id))
    ).is_ok()
    assert (
        await cache.get(
            None, tokens, destination, trace_context=context(request_id)
        )
    ).is_ok()
    events = records(cache, output)
    assert [e["event_type"] for e in events] == [
        "codec_encode",
        "l2_put",
        "l2_get",
        "codec_decode",
        "kv_import",
    ]
    assert all(e["block_key"] is None for e in events)
    assert events[1]["connector_write_bytes"] == sum(
        p.stat().st_size for p in root.rglob("*") if p.is_file()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,ctx", [(None, context()), ("DETAILED", context(mode=ProfileMode.OFF))]
)
async def test_off_never_constructs_events(setup_cache, mode, ctx, monkeypatch):
    create, _, output = setup_cache
    cache, tokens, source, destination = create(mode)
    if mode is None:
        assert isinstance(cache._profiler_recorder, NullRecorder)

    def forbidden(*args, **kwargs):
        pytest.fail("OFF constructed TraceEvent")

    monkeypatch.setattr(
        "aibrix_kvcache.compression_profiler.operation.TraceEvent", forbidden
    )
    assert (await cache.put(None, tokens, source, trace_context=ctx)).is_ok()
    assert (
        await cache.get(None, tokens, destination, trace_context=ctx)
    ).is_ok()
    cache._profiler_recorder.flush()
    assert not output.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["LIGHT", "DETAILED"])
async def test_corruption_rejected_without_destination_mutation(
    setup_cache, mode
):
    create, root, output = setup_cache
    cache, tokens, source, destination = create(mode)
    assert (
        await cache.put(None, tokens, source, trace_context=context())
    ).is_ok()
    for path in root.rglob("*"):
        if path.is_file():
            data = bytearray(path.read_bytes())
            data[-1] ^= 255
            path.write_bytes(data)
    before = [mr.tobytes() for mr in destination]
    assert (
        await cache.get(None, tokens, destination, trace_context=context())
    ).is_error()
    assert [mr.tobytes() for mr in destination] == before
    errors = [
        e for e in records(cache, output) if e["event_type"] == "error_recovery"
    ]
    assert len(errors) == 2
    assert all(e["result"] == "rejected" and e["error_class"] for e in errors)
    assert validate_trace(
        records(cache, output), cache._profiler_recorder.status
    ).valid


@pytest.mark.parametrize(
    "setting,value",
    [
        ("MODE", "UNKNOWN"),
        ("CAPACITY", "0"),
        ("CAPACITY", "abc"),
        ("CAPACITY", "-1"),
    ],
)
def test_invalid_config_fails_init(setup_cache, monkeypatch, setting, value):
    create, _, _ = setup_cache
    monkeypatch.setenv(ENV + setting, value)
    with pytest.raises(ValueError, match=ENV + setting):
        create(mode=None)


@pytest.mark.asyncio
async def test_concurrent_requests_keep_distinct_dependencies(
    setup_cache, monkeypatch
):
    create, _, output = setup_cache
    request_ids = [f"target-{index}" for index in range(8)]
    monkeypatch.setenv(ENV + "REQUEST_IDS", ",".join(request_ids))
    cache, tokens, source, _ = create()
    results = await asyncio.gather(
        *(
            cache.put(None, tokens, source, trace_context=context(request_id))
            for request_id in request_ids
        )
    )
    events = records(cache, output)
    errors = [e for e in events if e["event_type"] == "error_recovery"]
    assert len(events) - len(errors) == 32
    failed_stores = [e for e in events if e["event_type"] == "l2_put"
                     and e["result"] == "store_error"]
    assert len(errors) == len(failed_stores)
    # Concurrent same-key SHFS puts can contend on its existing .tmp path.
    failed_calls = sum(not result.is_ok() for result in results)
    assert failed_calls == len({e["request_id"] for e in errors})
    for request_id, result in zip(request_ids, results):
        request_errors = [e for e in errors if e["request_id"] == request_id]
        assert result.is_ok() == (len(request_errors) == 0)
    by_id = {e["operation_id"]: e for e in events}
    assert len(by_id) == len(events)
    for event in events:
        if event["event_type"] == "l2_put":
            dependency = by_id[event["dependency_ids"][0]]
            assert dependency["event_type"] == "codec_encode"
            assert dependency["block_key"] == event["block_key"]
    assert {e["dependency_ids"][0] for e in errors} == {
        e["operation_id"] for e in failed_stores
    }


@pytest.mark.asyncio
async def test_miss_reports_load_without_decode_or_restore(setup_cache):
    create, _, output = setup_cache
    cache, tokens, _, destination = create()
    before = [mr.tobytes() for mr in destination]
    assert (
        await cache.get(None, tokens, destination, trace_context=context())
    ).is_not_found()
    events = records(cache, output)
    assert len(events) == 2
    assert all(
        e["event_type"] == "l2_get"
        and e["result"] == "not_found"
        and e["connector_read_bytes"] is None
        for e in events
    )
    assert [mr.tobytes() for mr in destination] == before


@pytest.mark.asyncio
async def test_storage_failures_preserve_exception_contract(
    setup_cache, monkeypatch
):
    create, _, output = setup_cache
    cache, tokens, source, destination = create()
    failure = OSError("storage unavailable")

    async def rejected_put(*args):
        raise failure

    monkeypatch.setattr(cache._backend, "put_bytes", rejected_put)
    result = await cache.put(None, tokens, source, trace_context=context())
    assert result.is_error() and result.value is failure
    errors = [
        e for e in records(cache, output) if e["event_type"] == "error_recovery"
    ]
    assert len(errors) == 2 and all(
        e["error_class"] == "OSError" for e in errors
    )

    async def rejected_get(*args):
        raise failure

    monkeypatch.setattr(cache._backend, "get_bytes", rejected_get)
    # Existing get_bytes exceptions propagate through TaskGroup.
    with pytest.raises(ExceptionGroup) as raised:  # noqa: F821 (3.11 builtin)
        await cache.get(None, tokens, destination, trace_context=context())
    assert all(exc is failure for exc in raised.value.exceptions)
    errors = [
        e for e in records(cache, output) if e["event_type"] == "error_recovery"
    ]
    assert len(errors) == 4


@pytest.mark.asyncio
async def test_failed_put_status_does_not_claim_written_bytes(
    setup_cache, monkeypatch
):
    create, _, output = setup_cache
    cache, tokens, source, _ = create()
    failure = Status(StatusCodes.ERROR, value=OSError("storage full"))

    async def fail(*args):
        return failure

    monkeypatch.setattr(cache._backend, "put_bytes", fail)
    assert (
        await cache.put(None, tokens, source, trace_context=context())
        is failure
    )
    stores = [e for e in records(cache, output) if e["event_type"] == "l2_put"]
    assert all(
        e["connector_write_bytes"] is None and e["result"] == "store_error"
        for e in stores
    )


@pytest.mark.asyncio
async def test_export_failure_does_not_fail_storage(setup_cache, monkeypatch):
    create, _, output = setup_cache
    output.mkdir()
    cache, tokens, source, destination = create()
    assert (
        await cache.put(None, tokens, source, trace_context=context())
    ).is_ok()
    assert (
        await cache.get(None, tokens, destination, trace_context=context())
    ).is_ok()
    cache._profiler_recorder.flush()
    assert "export_error" in cache._profiler_recorder.status.reasons
    assert [mr.tobytes() for mr in destination] == [
        mr.tobytes() for mr in source
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["OFF", "LIGHT", "DETAILED"])
@pytest.mark.parametrize("batch", [False, True])
async def test_raw_storage_events_preserve_batching(
    setup_cache, monkeypatch, mode, batch
):
    create, _, output = setup_cache
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_MOCK_USE_MPUT_MGET", str(int(batch)))
    cache, tokens, source, destination = create(mode, codec="", backend="MOCK")
    assert cache._backend.feature.mput_mget == batch
    assert (
        await cache.put(None, tokens, source, trace_context=context())
    ).is_ok()
    assert (
        await cache.get(None, tokens, destination, trace_context=context())
    ).is_ok()
    assert [mr.tobytes() for mr in destination] == [
        mr.tobytes() for mr in source
    ]
    if mode == "OFF":
        assert not output.exists()
        return
    events = records(cache, output)
    assert len(events) == (4 if mode == "DETAILED" else 2)
    assert {e["event_type"] for e in events} == {"l2_put", "l2_get"}
    assert sum(e["connector_write_bytes"] or 0 for e in events) == sum(
        mr.length for mr in source
    )
    assert sum(e["connector_read_bytes"] or 0 for e in events) == sum(
        mr.length for mr in destination
    )
    assert all(
        (e["block_key"] is not None) == (mode == "DETAILED") for e in events
    )
    assert all(e["phase"] == ("batch" if batch else "") for e in events)


@pytest.mark.asyncio
async def test_mixed_light_bytes_remain_unknown(setup_cache, monkeypatch):
    create, _, output = setup_cache
    cache, tokens, source, _ = create("LIGHT")
    put = cache._backend.put_bytes
    calls = 0

    async def partial(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Status(StatusCodes.ERROR, value=OSError("partial write"))
        return await put(*args)

    monkeypatch.setattr(cache._backend, "put_bytes", partial)
    assert (
        await cache.put(None, tokens, source, trace_context=context())
    ).is_error()
    store = next(
        e for e in records(cache, output) if e["event_type"] == "l2_put"
    )
    assert store["result"] == "mixed"
    assert store["connector_write_bytes"] is None


@pytest.mark.asyncio
async def test_raw_shfs_bytes_and_background_serialization(
    setup_cache, monkeypatch
):
    create, root, output = setup_cache
    original = TraceEvent.to_record
    threads = []

    def serialize(event):
        threads.append(threading.current_thread().name)
        return original(event)

    original_clock = time.clock_gettime_ns

    def raw_clock(clock):
        assert clock == time.CLOCK_MONOTONIC_RAW
        return original_clock(clock)

    monkeypatch.setattr(TraceEvent, "to_record", serialize)
    monkeypatch.setattr(time, "clock_gettime_ns", raw_clock)
    cache, tokens, source, destination = create(codec="")
    assert (
        await cache.put(None, tokens, source, trace_context=context())
    ).is_ok()
    assert (
        await cache.get(None, tokens, destination, trace_context=context())
    ).is_ok()
    events = records(cache, output)
    assert sum(e["connector_write_bytes"] or 0 for e in events) == sum(
        p.stat().st_size for p in root.rglob("*") if p.is_file()
    )
    assert len(threads) == 4
    assert all(name.startswith("compression-trace-") for name in threads)
