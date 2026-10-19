# Copyright 2024 The Aibrix Team.
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
"""Hybrid-cache offloading connector for unmodified vLLM (>= 0.30).

Load with::

    --kv-transfer-config '{"kv_connector": "AIBrixHybridOffloadingConnector",
      "kv_connector_module_path":
        "aibrix_kvcache.integration.vllm.hybrid.connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {"backend": "MEMORY",
        "compression": "ANS", "namespace": "qwen-next"}}'

Only public connector APIs are used; nothing in vLLM is patched. A record
is the Mamba state at an aligned boundary plus attention spans down to 0;
a save writes only the span past the prefix's last saved boundary (see
``logic``). Restores are asynchronous: the request waits for
``get_finished`` so vLLM's ``preprocess_mamba`` (run before
``start_load_kv``) copies the already-restored boundary state.

Observability is the compression profiler (``AIBRIX_KV_CACHE_OL_COMPRESSION_
PROFILER_*``): every process writes ``<dir>/<scheduler|rankN>/<name>``. A
restore's EVICTION event is the scheduler seeing a local prefix miss for a
stored record; it spans any wait for an in-flight save and depends on each
rank's L2_PUT, linking the restore to the request that stored the record.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

from ....compression_profiler import EventType, ProfileMode, TraceContext
from ....compression_profiler.operation import (
    OperationTrace,
    TraceSink,
    open_recorder,
    request_context,
    request_event,
)
from ....l2.connectors.connector import Connector, ConnectorConfig
from .groups import describe_groups
from .layout import HybridLayout
from .logic import (
    GroupPlan,
    HybridIndex,
    HybridPlanError,
    Part,
    record_key,
)
from .metadata import (
    HybridMetadata,
    HybridWorkerMetadata,
    LoadJob,
    SaveJob,
)
from .store import HybridStore, HybridStoreError

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


class AIBrixHybridOffloadingConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._namespace = str(extra["namespace"])
        self._extra = extra
        self._kv_cache_config = kv_cache_config
        self._layers, kinds, sizes = describe_groups(kv_cache_config)
        align, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self._plan = GroupPlan(kinds, sizes, align, hash_block_size)
        self._world = vllm_config.parallel_config.tensor_parallel_size
        self._sink: Optional[TraceSink] = None
        if role == KVConnectorRole.SCHEDULER:
            self._sink = self._open_sink("scheduler", "")
        # Scheduler state.
        self._index = HybridIndex(world_size=self._world)
        self._requests: dict[str, "Request"] = {}
        # Request -> (block hash, boundary, span chain); boundary < 0: verify.
        self._pending: dict[str, tuple[bytes, int, list]] = {}
        self._ready: list[LoadJob] = []
        self._verify = bool(extra.get("verify_local_hits", False))
        # Diagnostic control: run the connector but never save records.
        self._save_enabled = bool(extra.get("save_enabled", True))
        self._verifies: list[LoadJob] = []
        self._saved_boundary: dict[str, int] = {}
        self._contexts: dict[str, TraceContext] = {}
        self._miss_since: dict[str, int] = {}
        # Waiting longer for an in-flight save than recomputing takes
        # gains nothing; a stuck save must not hold requests forever.
        self._save_wait_ns = int(float(extra.get("save_wait_s", 10)) * 1e9)
        # Worker state.
        self._store: Optional[HybridStore] = None
        self._rank = 0
        self._load_errors: set[int] = set()
        self._finished_recving: set[str] = set()
        self._worker_meta = HybridWorkerMetadata()
        # Background save completions append to _worker_meta.
        self._meta_lock = threading.Lock()

    def _open_sink(self, role: str, signature: str) -> Optional[TraceSink]:
        recorder, mode, requests = open_recorder(role)
        if mode == ProfileMode.OFF:
            return None
        groups = sorted(
            g for g, kind in self._plan.kinds.items() if kind != "skip"
        )
        return TraceSink(
            recorder,
            mode,
            requests,
            str(self._extra["compression"]).lower(),
            ",".join(map(str, groups)),
            signature,
        )

    def _trace(self, context) -> Optional[OperationTrace]:
        if self._sink is None or context is None:
            return None
        return OperationTrace(self._sink, context)

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------
    def _context(self, request: "Request") -> Optional[TraceContext]:
        req_id = request.request_id
        if req_id not in self._contexts and self._sink is not None:
            context = request_context(req_id)
            if context is not None:
                request_event(
                    self._sink.recorder,
                    context,
                    OperationTrace.now(),
                    phase="scheduled",
                    result="received",
                )
                self._contexts[req_id] = context
        return self._contexts.get(req_id)

    def _evicted(self, request, block_hash, boundary) -> tuple[str, ...]:
        """Record the local miss that sends a request to a stored record."""
        start = self._miss_since.pop(request.request_id, None)
        trace = self._trace(self._context(request))
        if trace is None or start is None:
            return ()
        with trace.observation():
            operation = trace.event(
                EventType.EVICTION,
                block_hash,
                start,
                self._index.puts(block_hash, boundary),
                phase="local_prefix_cache",
                result="local_miss",
            )
            trace.finish()
            return (operation,)
        return ()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        self._requests[request.request_id] = request
        self._context(request)
        hashes = [bytes(h) for h in request.block_hashes]
        now = OperationTrace.now()
        if not num_computed_tokens and now - self._miss_since.setdefault(
            request.request_id, now
        ) < self._save_wait_ns and self._index.pending_for(
            hashes, self._plan, num_tokens=request.num_tokens
        ):
            # A matching record is still being written; ask again next step.
            return None, False
        boundary = self._index.longest_hit(
            hashes, self._plan, num_tokens=request.num_tokens
        )
        # A local GPU prefix hit is served by vLLM; only cold requests load.
        if num_computed_tokens:
            self._miss_since.pop(request.request_id, None)
            if self._verify and boundary == num_computed_tokens:
                self._pending[request.request_id] = self._record(
                    hashes, -boundary
                )
            return 0, False
        if not boundary:
            self._miss_since.pop(request.request_id, None)
            return 0, False
        self._pending[request.request_id] = self._record(hashes, boundary)
        return boundary, True

    def _record(self, hashes, boundary) -> tuple[bytes, int, list]:
        block_hash = hashes[abs(boundary) // self._plan.hash_block_size - 1]
        chain = self._index.chain(block_hash, abs(boundary))
        assert chain is not None
        return block_hash, boundary, chain

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        pending = self._pending.pop(request.request_id, None)
        if pending is None or (num_external_tokens == 0 and pending[1] > 0):
            return
        block_hash, boundary, chain = pending
        context = self._contexts.get(request.request_id)
        if boundary < 0:
            job = LoadJob(
                request.request_id,
                block_hash,
                -boundary,
                self._plan.load_parts(
                    blocks.get_block_ids(), block_hash, -boundary, chain
                ),
                context,
            )
            self._verifies.append(job)
            return
        assert num_external_tokens == boundary
        # The record already exists; do not re-save it at this boundary.
        self._saved_boundary[request.request_id] = boundary
        self._ready.append(
            LoadJob(
                request.request_id,
                block_hash,
                boundary,
                self._plan.load_parts(
                    blocks.get_block_ids(), block_hash, boundary, chain
                ),
                context,
                self._evicted(request, block_hash, boundary),
            )
        )

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        meta = HybridMetadata(loads=self._ready, verifies=self._verifies)
        self._ready, self._verifies = [], []
        state = getattr(scheduler_output, "kv_connector_block_state", None)
        if state is None or not self._save_enabled:
            return meta
        for req_id, offloads in state.boundary_state_offloads.items():
            request = self._requests.get(req_id)
            if request is None:
                continue
            boundary = self._plan.save_boundary(
                offloads, prefill_end=request.num_prompt_tokens
            )
            if boundary is None or boundary <= self._saved_boundary.get(
                req_id, 0
            ):
                continue
            block_ids = state.get_block_ids(req_id)
            if block_ids is None:
                continue
            hashes = [bytes(h) for h in request.block_hashes]
            block_hash = hashes[boundary // self._plan.hash_block_size - 1]
            if self._index.has(block_hash, boundary) or self._index.saving(
                block_hash, boundary
            ):
                # Another request (e.g. a local prefix hit) already stored it.
                self._saved_boundary[req_id] = boundary
                continue
            # Only the span past the prefix's last saved boundary is new.
            parent = self._index.parent_for(
                hashes, self._plan, boundary=boundary
            )
            start = parent[1] if parent else 0
            try:
                parts = self._plan.save_parts(
                    block_ids, offloads, block_hash, boundary, start=start
                )
            except HybridPlanError as exc:
                logger.warning("[AIBrixHybrid] skip save %s: %s", req_id, exc)
                continue
            meta.saves.append(
                SaveJob(
                    req_id, block_hash, boundary, parts,
                    self._contexts.get(req_id),
                )
            )
            self._index.begin_save(block_hash, boundary, parent)
            self._saved_boundary[req_id] = boundary
        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, HybridWorkerMetadata):
            return
        for part, block_hash, boundary, start, rank, put in meta.stored:
            self._index.mark_stored(
                part, block_hash, boundary, rank=rank, start=start, put=put
            )
        for part, block_hash, boundary, start in meta.failed:
            self._index.mark_failed(part, block_hash, boundary, start=start)
        for part, block_hash, boundary, start in meta.save_failed:
            self._index.save_failed(part, block_hash, boundary, start=start)

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        for table in (
            self._requests,
            self._pending,
            self._saved_boundary,
            self._contexts,
            self._miss_since,
        ):
            table.pop(request.request_id, None)
        # wait_for_save enqueued each save's reads ahead of any block reuse.
        return False, None

    def request_finished(self, request: "Request", block_ids: list[int]):
        return self.request_finished_all_groups(request, (block_ids,))

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._rank = get_tensor_model_parallel_rank()
        assert get_tensor_model_parallel_world_size() == self._world
        layout = HybridLayout.build(
            self._layers,
            kv_caches,
            num_blocks=self._kv_cache_config.num_blocks,
        )
        backend = Connector.create(
            ConnectorConfig(
                backend_name=str(self._extra["backend"]),
                namespace=self._namespace,
                partition_id=f"tp{self._rank}",
                executor=ThreadPoolExecutor(2),
                layout_signature=layout.signature(),
            )
        )
        status = backend.open()
        if not status.is_ok():
            raise RuntimeError(f"AIBrix hybrid backend open failed: {status}")
        self._store = HybridStore(
            layout,
            backend,
            str(self._extra["compression"]),
            pin_records=self._extra["backend"] == "MEMORY",
        )
        self._sink = self._open_sink(f"rank{self._rank}", layout.signature())

    def _key(self, part: Part) -> bytes:
        return record_key(
            self._namespace,
            part.block_hash,
            part.end,
            rank=self._rank,
            part=part.part,
            start=part.start,
        )

    def _each_part(self, job: LoadJob, run) -> Optional[Part]:
        """``run(key, block_ids, trace)`` per part; the first failing part."""
        for part in job.parts:
            try:
                run(self._key(part), part.block_ids, self._trace(job.context))
            except HybridStoreError as exc:
                logger.warning("[AIBrixHybrid] %s: %s", job.req_id, exc)
                return part
        return None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs):
        meta = self._get_connector_metadata()
        assert isinstance(meta, HybridMetadata) and self._store is not None
        for job in meta.verifies:
            # The profiler records the per-slot comparison result.
            self._each_part(job, self._store.compare)
        for job in meta.loads:
            failed = self._each_part(
                job,
                lambda key, ids, trace, job=job: self._store.load(
                    key, ids, readback=self._verify, trace=trace,
                    after=job.after,
                ),
            )
            if failed is not None:
                # Recompute it all (kv_load_failure_policy=recompute).
                for part in job.parts:
                    for ids in part.block_ids.values():
                        self._load_errors.update(ids)
                self._worker_meta.failed.append(
                    (failed.part, failed.block_hash, failed.end, failed.start)
                )
            self._finished_recving.add(job.req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        """Stage each save synchronously; encode and store in background."""
        meta = self._get_connector_metadata()
        assert isinstance(meta, HybridMetadata) and self._store is not None
        for job in meta.saves:
            for part in job.parts:
                try:
                    future = self._store.save_async(
                        self._key(part),
                        part.block_ids,
                        self._trace(job.context),
                    )
                except (HybridStoreError, ValueError) as exc:
                    self._save_done(job, part, None, exc)
                    continue
                future.add_done_callback(
                    lambda f, job=job, part=part: self._save_done(
                        job,
                        part,
                        f.result() if not f.exception() else None,
                        f.exception(),
                    )
                )

    def _save_done(self, job, part, stats, error) -> None:
        # Index a part only once every byte is in the backend.
        if error is not None:
            logger.warning("[AIBrixHybrid] save %s: %s", job.req_id, error)
        with self._meta_lock:
            if error is None:
                self._worker_meta.stored.append((
                    part.part, job.block_hash, job.boundary, part.start,
                    self._rank, stats["put"],
                ))
            else:
                self._worker_meta.save_failed.append(
                    (part.part, job.block_hash, job.boundary, part.start)
                )

    def shutdown(self):
        if self._store is not None:
            self._store.close()
        if self._sink is not None:
            self._sink.recorder.close()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        done, self._finished_recving = self._finished_recving, set()
        return None, done or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        errors, self._load_errors = self._load_errors, set()
        return errors

    def build_connector_worker_meta(
        self,
    ) -> Optional[KVConnectorWorkerMetadata]:
        with self._meta_lock:
            meta, self._worker_meta = self._worker_meta, HybridWorkerMetadata()
        if not meta.stored and not meta.failed and not meta.save_failed:
            return None
        return meta
