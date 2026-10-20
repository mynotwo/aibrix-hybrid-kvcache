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

from types import SimpleNamespace

import pytest

from aibrix_kvcache.compression_profiler import ProfileMode
from aibrix_kvcache.l2.l2_cache import _L2Trace
from aibrix_kvcache.status import Status, StatusCodes

from .test_compression_profiler_l2 import context, records
from .test_compression_profiler_l2 import setup_cache as setup_cache


def broken(*args, **kwargs):
    raise RuntimeError("observation failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["put", "get"])
async def test_codec_failure_survives_broken_observer(
    setup_cache, monkeypatch, operation
):
    create, _, _ = setup_cache
    cache, tokens, source, destination = create()
    off = context(mode=ProfileMode.OFF)
    if operation == "get":
        assert (
            await cache.put(None, tokens, source, trace_context=off)
        ).is_ok()
    failure = ValueError("real codec failure")

    def failed_codec(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        cache._compression,
        "encode" if operation == "put" else "decode",
        failed_codec,
    )
    monkeypatch.setattr(_L2Trace, "event", broken)
    before = [mr.tobytes() for mr in destination]
    result = await getattr(cache, operation)(
        None,
        tokens,
        source if operation == "put" else destination,
        trace_context=context(),
    )
    assert result.is_error() and result.value is failure
    assert [mr.tobytes() for mr in destination] == before
    assert not cache._profiler_recorder.status.valid


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["LIGHT", "DETAILED"])
@pytest.mark.parametrize("operation", ["put", "get"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "fault", ["clock", "event", "record", "finish", "storage"]
)
async def test_raw_observer_failure_preserves_storage(
    setup_cache, monkeypatch, mode, operation, batch, fault
):
    create, _, _ = setup_cache
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_MOCK_USE_MPUT_MGET", str(int(batch)))
    cache, tokens, source, destination = create(mode, codec="", backend="MOCK")
    off = context(mode=ProfileMode.OFF)
    if operation == "get":
        assert (
            await cache.put(None, tokens, source, trace_context=off)
        ).is_ok()
    target = cache._profiler_recorder if fault == "record" else _L2Trace
    with monkeypatch.context() as patch:
        patch.setattr(target, "now" if fault == "clock" else fault, broken)
        result = await getattr(cache, operation)(
            None,
            tokens,
            source if operation == "put" else destination,
            trace_context=context(),
        )
    assert result.is_ok()
    assert not cache._profiler_recorder.status.valid
    if operation == "put":
        assert (
            await cache.get(None, tokens, destination, trace_context=off)
        ).is_ok()
    assert [mr.tobytes() for mr in destination] == [
        mr.tobytes() for mr in source
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["LIGHT", "DETAILED"])
@pytest.mark.parametrize("operation", ["put", "get"])
@pytest.mark.parametrize(
    "fault", ["clock", "sizes", "event", "record", "finish"]
)
async def test_observation_failure_preserves_storage(
    setup_cache, monkeypatch, mode, operation, fault
):
    create, _, _ = setup_cache
    cache, tokens, source, destination = create(mode)
    off = context(mode=ProfileMode.OFF)
    if operation == "get":
        assert (
            await cache.put(None, tokens, source, trace_context=off)
        ).is_ok()
    with monkeypatch.context() as patch:
        target, name = {
            "clock": (_L2Trace, "now"),
            "sizes": (cache._compression, "encoded_sizes"),
            "event": (_L2Trace, "event"),
            "record": (cache._profiler_recorder, "record"),
            "finish": (_L2Trace, "finish"),
        }[fault]
        patch.setattr(target, name, broken)
        regions = source if operation == "put" else destination
        result = await getattr(cache, operation)(
            None, tokens, regions, trace_context=context()
        )
    assert result.is_ok(), result
    assert not cache._profiler_recorder.status.valid
    assert "observation_error" in cache._profiler_recorder.status.reasons
    if operation == "put":
        assert (
            await cache.get(None, tokens, destination, trace_context=off)
        ).is_ok()
    assert [mr.tobytes() for mr in destination] == [
        mr.tobytes() for mr in source
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["LIGHT", "DETAILED"])
@pytest.mark.parametrize(
    "codec,batch", [("ZSTD", False), ("", False), ("", True)]
)
@pytest.mark.parametrize("operation", ["put", "get"])
async def test_failed_status_keeps_individual_error_events(
    setup_cache, monkeypatch, mode, codec, batch, operation
):
    create, _, output = setup_cache
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_MOCK_USE_MPUT_MGET", str(int(batch)))
    cache, tokens, source, destination = create(
        mode, codec=codec, backend="SHFS" if codec else "MOCK"
    )
    failure = Status(StatusCodes.ERROR, value=OSError("disk unavailable"))

    async def rejected(*args):
        return [failure, failure] if batch else failure

    method = (
        operation + "_bytes" if codec else ("m" if batch else "") + operation
    )
    monkeypatch.setattr(cache._backend, method, rejected)
    regions = source if operation == "put" else destination
    result = await getattr(cache, operation)(
        None, tokens, regions, trace_context=context()
    )
    # Preserve pre-existing public mget mapping to NOT_FOUND; its underlying
    # per-block ERROR statuses must still be observable independently.
    if batch and operation == "get":
        assert result.is_not_found()
    else:
        assert result is failure
    events = records(cache, output)
    errors = [e for e in events if e["event_type"] == "error_recovery"]
    assert len(errors) == 2
    assert all(
        e["result"] == "rejected" and e["error_class"] == "OSError"
        for e in errors
    )
    assert len({e["operation_id"] for e in errors}) == 2
    by_id = {e["operation_id"]: e for e in events}
    assert all(
        by_id[e["dependency_ids"][0]]["event_type"] == "l2_" + operation
        for e in errors
    )


@pytest.mark.asyncio
async def test_raw_mput_returns_first_failed_status(setup_cache, monkeypatch):
    create, _, _ = setup_cache
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_MOCK_USE_MPUT_MGET", "1")
    cache, tokens, source, _ = create("OFF", codec="", backend="MOCK")
    failure = Status(StatusCodes.ERROR, value=OSError("disk unavailable"))

    async def rejected(*args):
        return [failure, Status(StatusCodes.ERROR, value=OSError("other"))]

    monkeypatch.setattr(cache._backend, "mput", rejected)
    assert await cache.put(None, tokens, source) is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("codec", ["IDENTITY", "ZSTD"])
async def test_l2_accounting_fast_path_has_provenance_and_no_json_copy(
    setup_cache, monkeypatch, codec
):
    create, root, _ = setup_cache
    cache, tokens, source, destination = create(codec=codec)
    adapter = cache._compression
    serialize, deserialize, sizes = (
        adapter.serialize,
        adapter.deserialize,
        adapter.encoded_sizes,
    )
    provenance = {}
    counted = []

    class NoSlice(bytes):
        def __getitem__(self, item):
            if isinstance(item, slice):
                raise AssertionError("accounting copied bytes")
            return super().__getitem__(item)

    def encode_envelope(block):
        envelope = serialize(block)
        provenance[id(envelope)] = block
        return envelope

    def decode_envelope(envelope):
        block = deserialize(envelope)
        provenance[id(envelope)] = block
        return block

    def checked_sizes(block, envelope, *, trusted=False):
        assert trusted
        assert provenance[id(envelope)] is block
        with monkeypatch.context() as patch:
            patch.setattr(
                "aibrix_kvcache.compression.adapter.json",
                SimpleNamespace(loads=broken, dumps=broken),
            )
            result = sizes(block, NoSlice(envelope), trusted=trusted)
        counted.append(result)
        return result

    monkeypatch.setattr(adapter, "serialize", encode_envelope)
    monkeypatch.setattr(adapter, "deserialize", decode_envelope)
    monkeypatch.setattr(adapter, "encoded_sizes", checked_sizes)
    assert (
        await cache.put(None, tokens, source, trace_context=context())
    ).is_ok()
    assert (
        await cache.get(None, tokens, destination, trace_context=context())
    ).is_ok()
    assert len(counted) == 4
    assert cache._profiler_recorder.status.valid
    for path in root.rglob("*"):
        if path.is_file():
            contents = bytearray(path.read_bytes())
            contents[-1] ^= 255
            path.write_bytes(contents)
    before = [mr.tobytes() for mr in destination]
    assert (
        await cache.get(None, tokens, destination, trace_context=context())
    ).is_error()
    assert len(counted) == 4
    assert [mr.tobytes() for mr in destination] == before
