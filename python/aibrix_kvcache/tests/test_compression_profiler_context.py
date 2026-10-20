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

"""Request identity at real cache boundaries and isolated connector methods.

Connector methods are compiled from production AST to avoid requiring CUDA or
vLLM at collection time. Only the imported engine/GPU collaborators are fakes.
"""

import ast
import asyncio
import enum
import json
import logging
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import aibrix_kvcache
from aibrix_kvcache import BaseKVCacheManager, KVCacheConfig, ModelSpec
from aibrix_kvcache.cache_hashable import TokenListView
from aibrix_kvcache.compression_profiler import (
    EventType,
    ProfileMode,
    TraceContext,
    TraceEvent,
)
from aibrix_kvcache.compression_profiler.validation import validate_trace
from aibrix_kvcache.status import Status
from aibrix_kvcache.utils import cpu_perf_timer, round_down

from .conftest import get_cache_conf

ENV = "AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_"
CONNECTORS = (
    Path(__file__).parents[1] / "aibrix_kvcache/integration/vllm/kv_connector"
)


def context(request="req-a"):
    return TraceContext("experiment", "run", request, ProfileMode.DETAILED)


@pytest.fixture
def cache_factory(monkeypatch, tmp_path):
    caches = []
    monkeypatch.setattr(
        "aibrix_kvcache.cache_manager.TESTING_DISABLE_PIN_MEMORY", True
    )
    settings = {
        "L1_CACHE_ENABLED": "0",
        "L1_CACHE_DEVICE": "cpu",
        "L1_CACHE_PIN_MEMORY": "0",
        "L1_CACHE_CAPACITY_GB": "0.001",
        "L2_CACHE_BACKEND": "MOCK",
        "L2_CACHE_INGESTION_TYPE": "ALL",
        "L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS": "0",
        "COMPRESSION_PROFILER_MODE": "DETAILED",
        "COMPRESSION_PROFILER_REQUEST_IDS": "req-a,req-b",
    }
    for key, value in settings.items():
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_" + key, value)
    _, spec = get_cache_conf(aibrix_kvcache.KVCacheBlockLayout.LCND)

    def create(
        l1=False, async_put=False, mode="DETAILED",
        manager=BaseKVCacheManager, **kwargs,
    ):
        output = tmp_path / f"events-{len(caches)}.jsonl"
        monkeypatch.setenv(ENV + "OUTPUT", str(output))
        monkeypatch.setenv(ENV + "MODE", mode)
        monkeypatch.setenv("AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED", str(int(l1)))
        monkeypatch.setenv(
            "AIBRIX_KV_CACHE_OL_L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS",
            "1024" if async_put else "0",
        )
        cache = manager(KVCacheConfig(spec, ModelSpec(1024)), **kwargs)
        caches.append(cache)
        return cache, TokenListView(list(range(spec.block_ntokens))), output

    yield create
    for cache in caches:
        cache.close()


def records(cache, output):
    cache._l2_cache._profiler_recorder.flush()
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_two_requests_same_block_do_not_share_operation_identity(cache_factory):
    cache, tokens, output = cache_factory()
    for request in ("req-a", "req-b"):
        handle = cache.allocate_for(None, tokens).get()
        assert cache.put(
            None, tokens, handle, trace_context=context(request)
        ).is_ok()
        status = cache.acquire(None, tokens, trace_context=context(request))
        assert status.is_ok()
        status.get()[1].release()
    events = records(cache, output)
    assert {e["request_id"] for e in events} == {"req-a", "req-b"}
    assert len({e["operation_id"] for e in events}) == len(events)
    assert {e["result"] for e in events} >= {"hit", "stored"}
    assert all(e["generation"] is None for e in events)


def test_executor_completion_closes_original_request_span(
    cache_factory, monkeypatch
):
    cache, tokens, output = cache_factory(async_put=True)
    entered, release = threading.Event(), threading.Event()
    original = cache._l2_cache.put

    async def delayed(*args, **kwargs):
        entered.set()
        await asyncio.get_running_loop().run_in_executor(None, release.wait)
        return await original(*args, **kwargs)

    monkeypatch.setattr(cache._l2_cache, "put", delayed)
    handle = cache.allocate_for(None, tokens).get()
    try:
        assert cache.put(None, tokens, handle, trace_context=context()).is_ok()
        assert entered.wait(5)
        other = cache.acquire(None, tokens, trace_context=context("req-b"))
        assert other.is_not_found()
    finally:
        release.set()
        cache.flush()
    stores = [e for e in records(cache, output) if e["event_type"] == "l2_put"]
    assert stores and {e["request_id"] for e in stores} == {"req-a"}


def connector_namespace(kind=1):
    source = CONNECTORS / f"aibrix_offloading_connector_type{kind}.py"
    tree = ast.parse(source.read_text())
    names = {
        "_request_trace_context",
        "AIBrixOffloadingConnectorRequestState",
        "AIBrixOffloadingConnectorRequestMetadata",
        "AIBrixOffloadingConnectorMetadata",
        "AIBrixOffloadingConnectorWorker",
    }
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.bases = []
            node.body = [
                method
                for method in node.body
                if getattr(method, "name", "") != "__del__"
            ]
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = []
    namespace = dict(
        enum=enum,
        dataclass=dataclass,
        field=field,
        torch=torch,
        uuid=uuid,
        time=time,
        aibrix_kvcache=aibrix_kvcache,
        TraceContext=TraceContext,
        ProfileMode=ProfileMode,
        EventType=EventType,
        round_down=round_down,
        logger=logging.getLogger(__name__),
        logging=logging,
        OFFLOADING_CONNECTOR_SKIP_THRESHOLD=0,
        log_if=Mock(),
        log_every_n_seconds=Mock(),
        perf_timer=cpu_perf_timer,
        reshape_and_cache_multi_layer=Mock(),
        reshape_and_offload_multi_layer=Mock(),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(source), "exec"),
        namespace,
    )
    return namespace


def test_generation_update_and_retry_have_distinct_operation_ids(monkeypatch):
    monkeypatch.setenv(ENV + "MODE", "DETAILED")
    monkeypatch.setenv(ENV + "EXPERIMENT_ID", "matrix-experiment")
    monkeypatch.setenv(ENV + "RUN_ID", "matrix-run")
    ns = connector_namespace()
    metadata = ns["AIBrixOffloadingConnectorMetadata"]({})
    waiting = ns["AIBrixOffloadingConnectorRequestState"].WAITING_FOR_RECV
    metadata.upsert_request("req-a", state=waiting)
    first = metadata["req-a"].trace_context
    metadata.upsert_request(
        "req-a", state=waiting, resumed_from_preemption=True
    )
    retry = metadata["req-a"].trace_context
    metadata.upsert_request("req-a", state=waiting, context_len=16)
    update = metadata["req-a"].trace_context
    assert {first.request_id, retry.request_id, update.request_id} == {"req-a"}
    assert {first.experiment_id, retry.experiment_id, update.experiment_id} == {
        "matrix-experiment"
    }
    assert {first.run_id, retry.run_id, update.run_id} == {"matrix-run"}
    assert (
        len({first.operation_id, retry.operation_id, update.operation_id}) == 3
    )
    assert retry.dependency_ids == (first.operation_id,)
    assert update.dependency_ids == (retry.operation_id,)


def test_request_context_is_not_created_without_run_identity(monkeypatch):
    monkeypatch.setenv(ENV + "MODE", "LIGHT")
    monkeypatch.delenv(ENV + "EXPERIMENT_ID", raising=False)
    monkeypatch.delenv(ENV + "RUN_ID", raising=False)
    context = connector_namespace()["_request_trace_context"]("request-x")
    assert context is None


def test_l1_callback_owns_original_context_and_clears_stale_binding(
    cache_factory,
):
    cache, tokens, output = cache_factory(l1=True)
    handle = cache.allocate_for(None, tokens).get()
    mr = handle.memory_regions[0]
    mr._profiler_context = context("old")
    assert cache.put(None, tokens, handle, trace_context=context()).is_ok()
    assert mr._profiler_context.request_id == "req-a"
    assert {
        e["request_id"]
        for e in records(cache, output)
        if e["event_type"] == "l2_put"
    } == {"req-a"}
    # Reuse the exact object under the existing refcount ownership contract.
    mr.ref_up()
    assert cache._l1_cache.put(None, tokens, [mr]).is_ok()
    assert mr._profiler_context is None


def test_off_and_observation_errors_preserve_cache_results(
    cache_factory, monkeypatch
):
    cache, tokens, output = cache_factory(mode="OFF")
    monkeypatch.setattr(
        "aibrix_kvcache.cache_manager.TraceEvent",
        Mock(side_effect=AssertionError("constructed")),
    )
    assert cache.acquire(None, tokens, trace_context=context()).is_not_found()
    assert not output.exists()
    cache, tokens, output = cache_factory()
    assert cache.acquire(None, tokens, trace_context=context()).is_not_found()
    cache._l2_cache._profiler_recorder.flush()
    assert (
        "observation_error" in cache._l2_cache._profiler_recorder.status.reasons
    )


def test_layer_observation_is_explicit_scalar():
    event = TraceEvent("e", "r", "req-a", "op", EventType.KV_IMPORT)
    assert event.to_record()["layer_observed"] is False
    assert event.layer is None


def worker_fixture(kind, cache, tokens, monkeypatch):
    ns = connector_namespace(kind)
    worker = ns["AIBrixOffloadingConnectorWorker"].__new__(
        ns["AIBrixOffloadingConnectorWorker"]
    )
    worker.cache = cache
    worker.cache_block_ntokens = worker.engine_block_ntokens = len(tokens)
    worker.block_layout = cache.block_layout
    worker.kv_layout_blocks_first = False
    worker.layers_kv_caches = [torch.zeros(1)] * 2
    worker.k_scales = worker.v_scales = [1.0] * 2
    worker.no_compile_layers = {"layer.4": None, "layer.9": None}
    worker.layer_name_idx_mapping = {"layer.4": 0, "layer.9": 1}
    worker.layer_ids = [4, 9]
    worker.num_layers = 2
    worker._metrics = SimpleNamespace(time_measurement_enabled=False)
    worker._meta_cache = {
        "req-a": SimpleNamespace(
            get_context_tokens_view=lambda: tokens,
            context_slot_mapping=torch.arange(len(tokens)),
        )
    }
    worker._send_lengths = {"req-a": (0, len(tokens))}
    worker._allocated_kvcache_handles = {}
    worker._acquired_kvcache_handles = {}
    worker._import_spans = []
    worker._export_spans = []
    worker._save_contexts = {}
    worker._load_contexts = {}
    worker._export_dependencies = {}
    worker._recv_stream = worker._send_stream = Mock()
    worker._send_slot_mapping = worker._recv_slot_mapping = torch.zeros(
        len(tokens), dtype=torch.long
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    meta = SimpleNamespace(
        req_id="req-a",
        trace_context=context().child("initial"),
        prompt_len=len(tokens),
        context_len=0,
        query_len=len(tokens),
    )
    return worker, meta, ns


def test_type1_sync_transfer_keeps_unknown_layer(cache_factory, monkeypatch):
    cache, tokens, output = cache_factory()
    worker, meta, ns = worker_fixture(1, cache, tokens, monkeypatch)
    worker._send_kv_sync_impl(meta)
    assert worker._recv_kv_sync_impl(meta) == len(tokens)
    events = records(cache, output)
    transfers = [
        e for e in events if e["event_type"] in ("kv_import", "kv_export")
    ]
    assert {e["event_type"] for e in transfers} == {"kv_import", "kv_export"}
    assert all(
        e["request_id"] == meta.req_id
        and e["layer"] is None
        and not e["layer_observed"]
        for e in transfers
    )
    assert ns["reshape_and_cache_multi_layer"].call_count == 1
    assert ns["reshape_and_offload_multi_layer"].call_count == 1


def test_type2_layer_completion_uses_submission_attempt(
    cache_factory, monkeypatch
):
    cache, tokens, output = cache_factory()
    worker, meta, _ = worker_fixture(2, cache, tokens, monkeypatch)
    metadata = {"req-a": meta}
    worker.save_kv_layer(metadata, "layer.4", torch.zeros(1), None)
    worker.save_kv_layer(metadata, "layer.9", torch.zeros(1), None)
    initial = meta.trace_context
    meta.trace_context = initial.child(
        "retry", dependencies=(initial.operation_id,)
    )
    worker.wait_for_save(metadata)
    assert worker._recv_kv_impl(meta) == len(tokens)
    worker._start_load_kv(metadata, "layer.4")
    meta.trace_context = meta.trace_context.child("later-update")
    worker.wait_for_layer_load(metadata, "layer.4")
    worker.wait_for_layer_load(metadata, "layer.9")
    events = records(cache, output)
    exports = [e for e in events if e["event_type"] == "kv_export"]
    imports = [e for e in events if e["event_type"] == "kv_import"]
    assert {e["layer"] for e in exports} == {4, 9}
    assert {e["layer"] for e in imports} == {4, 9}
    assert all(e["layer_observed"] for e in exports + imports)
    assert all(e["parent_operation_id"] == "initial" for e in exports)
    assert all(e["parent_operation_id"] == "retry" for e in imports)
    cache_put = next(e for e in events if e["phase"] == "cache_put")
    assert cache_put["parent_operation_id"] == "initial"
    assert set(cache_put["dependency_ids"]) == {
        e["operation_id"] for e in exports
    }


@pytest.mark.parametrize("ingestion", ["HOT", "EVICTED"])
@pytest.mark.parametrize("external", [False, True])
def test_l1_delayed_ingestion_keeps_producer_identity(
    cache_factory, monkeypatch, ingestion, external
):
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_L2_CACHE_INGESTION_TYPE", ingestion)
    cache, tokens, output = cache_factory(l1=True)
    handle = cache.allocate_for(None, tokens).get()
    if external:
        tensors = [t.clone() for t in handle.to_tensors()]
        handle.release()
        # Exercise L1's allocation/copy path, used for external input handles.
        assert cache._l1_cache.put(
            None, tokens, tensors, trace_context=context()
        ).is_ok()
    else:
        assert cache.put(None, tokens, handle, trace_context=context()).is_ok()
    if ingestion == "HOT":
        result = cache.acquire(None, tokens, trace_context=context("req-b"))
        result.get()[1].release()
    else:
        cache._l1_cache._eviction_policy.evict(cache.block_nbytes)
    stores = [e for e in records(cache, output) if e["event_type"] == "l2_put"]
    assert stores and {e["request_id"] for e in stores} == {"req-a"}


def test_partial_hit_reports_remaining_fallback(cache_factory):
    cache, tokens, output = cache_factory()
    handle = cache.allocate_for(None, tokens).get()
    cache.put(None, tokens, handle, trace_context=context())
    longer = TokenListView(list(range(2 * len(tokens))))
    result = cache.acquire(None, longer, trace_context=context("req-b"))
    assert result.is_ok() and result.get()[0] == len(tokens)
    result.get()[1].release()
    events = [e for e in records(cache, output) if e["request_id"] == "req-b"]
    fallback = next(e for e in events if e["event_type"] == "fallback")
    assert fallback["token_count"] == len(tokens)
    assert fallback["restore_result"] == "not_restored"


def test_zero_copy_acquire_receives_explicit_context(
    cache_factory, monkeypatch
):
    cache, tokens, _ = cache_factory()
    seen = []

    async def acquire(prefix, query, *, trace_context=None, trace_result=None):
        seen.append(trace_context)
        return Status.ok([])

    monkeypatch.setattr(cache._l2_cache, "acquire", acquire)
    cache._l2_acquire_impl(None, tokens, trace_context=context())
    assert seen == [context()]


def test_scheduler_context_to_worker_trace_has_complete_dependencies(
    cache_factory, monkeypatch
):
    monkeypatch.setenv(ENV + "EXPERIMENT_ID", "context-test")
    monkeypatch.setenv(ENV + "RUN_ID", "scheduler-worker-run")
    cache, tokens, output = cache_factory()
    worker, _, ns = worker_fixture(1, cache, tokens, monkeypatch)
    metadata = ns["AIBrixOffloadingConnectorMetadata"]({})
    state = ns["AIBrixOffloadingConnectorRequestState"].WAITING_FOR_RECV
    cached = worker._meta_cache["req-a"]
    cached.extend = Mock()
    for attempt in range(2):
        metadata.upsert_request(
            "req-a",
            state=state,
            prompt_len=len(tokens),
            context_len=0,
            query_len=len(tokens),
            resumed_from_preemption=bool(attempt),
        )
        worker._update_meta_cache(metadata)
        worker._send_kv_sync_impl(metadata["req-a"])
        worker._recv_kv_sync_impl(metadata["req-a"])
    events = records(cache, output)
    assert validate_trace(
        events, cache._l2_cache._profiler_recorder.status
    ).valid
    roots = [e for e in events if e["phase"] == "scheduled"]
    assert len(roots) == 2
    assert roots[1]["dependency_ids"] == [roots[0]["operation_id"]]


@pytest.mark.asyncio
async def test_zero_copy_l2_miss_retains_context(cache_factory, monkeypatch):
    cache, tokens, output = cache_factory()

    async def absent(key):
        from aibrix_kvcache.status import StatusCodes

        return Status(StatusCodes.NOT_FOUND)

    monkeypatch.setattr(cache._l2_cache._backend, "acquire", absent)
    result = await cache._l2_cache.acquire(
        None, tokens, trace_context=context()
    )
    assert result.is_not_found()
    event = records(cache, output)[0]
    assert event["request_id"] == "req-a"
    assert event["phase"] == "zero_copy_acquire"
    assert event["result"] == "not_found"


def test_type2_failed_stream_preserves_error_and_rejects_restore(
    cache_factory, monkeypatch
):
    cache, tokens, output = cache_factory()
    worker, meta, _ = worker_fixture(2, cache, tokens, monkeypatch)
    span = cache.trace_start(meta.trace_context)
    worker._import_spans = [(span, 4, len(tokens))]
    failure = RuntimeError("copy failed")
    worker._recv_stream.synchronize.side_effect = failure
    with pytest.raises(RuntimeError) as raised:
        worker.wait_for_layer_load({"req-a": meta}, "layer.4")
    assert raised.value is failure
    event = records(cache, output)[0]
    assert event["operation_id"] == span[0].operation_id
    assert event["restore_result"] == "failed"
    assert event["error_class"] == "RuntimeError"
    assert worker._import_spans == []
