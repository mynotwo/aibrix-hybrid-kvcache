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
import logging
import time
from concurrent.futures import Executor
from typing import Any, Callable, Iterator, List, Sequence, Tuple, cast

import torch
from more_itertools import batched

from ..cache_handle import KVCacheHandle
from ..cache_hashable import BlockHashes, KVCacheKeyTypes, TokenListView
from ..common.absl_logging import getLogger, log_every_n_seconds
from ..compression import IdentityCompressionAdapter, ZstdCompressionAdapter
from ..compression_profiler import (
    EventType,
    ProfileMode,
    TraceContext,
)
from ..compression_profiler.operation import (
    OperationTrace,
    TraceSink,
)
from ..compression_profiler.operation import (
    profiler_config as _profiler_config,
)
from ..compression_profiler.recorder import NullRecorder, ShardedRingRecorder
from ..memory import ExternalMemoryRegion, ManagedMemoryRegion, MemoryRegion
from ..meta_service import MetaService
from ..metrics import L2CacheMetrics, MeasurableBase, MetricRecorder
from ..profiling import nvtx_range
from ..spec import KVCacheBlockLayout, KVCacheBlockSpec
from ..status import Status, StatusCodes
from ..utils import buffer_to_tensor
from .connectors import (
    Connector,
    ConnectorConfig,
    ConnectorFeature,
    ConnectorZeroCopyMemoryRegion,
)
from .key_builders import KeyBuilder, RawKeyBuilder
from .placement import Placement, PlacementConfig

logger = getLogger(__name__)


class _L2Trace(OperationTrace):
    def __init__(self, cache, context: TraceContext, result=None):
        super().__init__(
            TraceSink(
                recorder=cache._profiler_recorder,
                mode=cache._profiler_mode,
                requests=cache._profiler_requests,
                codec=cache._codec,
                cache_group=cache._cache_group,
                cache_spec=cache.block_spec.signature,
            ),
            context,
            result,
        )

    def storage(self, kind, keys, mrs, statuses, start):
        if isinstance(statuses, Status):
            statuses = [statuses] * len(keys)
        for key, mr, status in zip(keys, mrs, statuses):
            length = (
                mr.length
                if isinstance(mr, MemoryRegion)
                else sum(region.length for region in mr)
            )
            writing = kind == EventType.L2_PUT
            values = {
                "connector_write_bytes"
                if writing
                else "connector_read_bytes": length if status.is_ok() else None,
                "destination_tier" if writing else "source_tier": "L2",
            }
            operation = self.event(
                kind,
                key,
                start,
                self.context.dependency_ids,
                phase="batch" if len(keys) > 1 else "",
                result=("stored" if writing else "loaded")
                if status.is_ok()
                else "not_found"
                if status.is_not_found()
                else "storage_error",
                **values,
            )
            self.failed_status(status, key, start, (operation,))

    def failed_status(self, status, key, start, dependencies):
        if status.is_ok() or status.is_not_found():
            return
        self.event(
            EventType.ERROR_RECOVERY, key, start, dependencies,
            result="rejected", restore_result="rejected",
            error_class=(type(status.value).__name__
                         if isinstance(status.value, Exception)
                         else status.error_code.name),
        )


class L2Cache(MeasurableBase):
    def __init__(
        self,
        backend_name: str,
        placement_policy: str,
        namespace: str,
        block_spec: KVCacheBlockSpec,
        executor: Executor,
        refresh_interval_s: int = 0,
        op_batch: int = 8,
        metrics: L2CacheMetrics | None = None,
        meta_service: MetaService | None = None,
        key_builder: KeyBuilder | None = None,
        compression: str = "",
    ) -> None:
        """Create a cache object.
        Args:
            backend_name (str): The name of cache backend.
            placement_policy (str): The placement policy.
            namespace (str): Namespace.
            block_spec (KVCacheBlockSpec): The block spec.
            executor (Executor): The executor.
            refresh_interval_s (int): The refresh interval in seconds.
            op_batch (int): The number of ops in a batch.
            metrics (L2CacheMetrics): metrics recorder.
            meta_service (MetaService): meta service.
            key_builder (KeyBuilder): key builder.
        """
        super().__init__(metrics)
        self.block_spec: KVCacheBlockSpec = block_spec
        self.block_layout: KVCacheBlockLayout = self.block_spec.block_layout
        self.block_shape: Tuple[int, ...] = self.block_spec.block_shape
        self.block_dtype: torch.dtype = self.block_spec.block_dtype
        self.block_ntokens: int = self.block_spec.block_ntokens
        self.block_nbytes: int = self.block_spec.block_nbytes
        self.block_shape_token_dim: int = self.block_spec.block_shape_token_dim
        self.key_builder: KeyBuilder = key_builder or RawKeyBuilder(
            self.block_ntokens
        )
        self.op_batch: int = op_batch
        self._executor: Executor = executor
        self._backend: Connector = None  # type: ignore
        self._use_compact_layout: bool = MemoryRegion.use_compact_layout()
        self._mr_ops_queue: asyncio.Queue | None = None
        self._mr_ops_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        compression = compression.strip().upper()
        if compression not in ("", "ZSTD", "IDENTITY"):
            raise ValueError(f"Unsupported L2 compression codec: {compression}")
        self._compression = (
            ZstdCompressionAdapter()
            if compression == "ZSTD"
            else IdentityCompressionAdapter()
            if compression == "IDENTITY"
            else None
        )
        self._codec = compression.lower() or None
        self._profiler_recorder = NullRecorder()
        mode, capacity, output, requests = _profiler_config()
        self._profiler_mode, self._profiler_requests = mode, requests

        cat_head_ids = "_".join(
            [
                str(self.block_spec.tensor_spec.heads[0]),
                str(self.block_spec.tensor_spec.heads[-1]),
            ]
        )
        cat_layer_ids = "_".join(
            [
                str(self.block_spec.tensor_spec.layers[0]),
                str(self.block_spec.tensor_spec.layers[-1]),
            ]
        )
        partition_id = f"h{cat_head_ids}_l{cat_layer_ids}"
        block_spec_signature = self.block_spec.signature
        key_builder_signature = self.key_builder.signature
        layout_signature = "c" if self._use_compact_layout else "ex"
        backend_config = ConnectorConfig(
            backend_name=backend_name,
            namespace=namespace,
            partition_id=partition_id,
            executor=executor,
            block_spec_signature=block_spec_signature,
            key_builder_signature=key_builder_signature,
            layout_signature=layout_signature,
        )

        if meta_service is None:
            # direct mode
            self._backend = Connector.create(backend_config)
        else:
            # cluster mode, using placement policy
            placement_config = PlacementConfig(
                placement_policy=placement_policy,
                conn_config=backend_config,
                meta_service=meta_service,
                refresh_interval_s=refresh_interval_s,
            )
            self._backend = Placement.create(placement_config)

        if (
            self._compression is not None
            and not self._backend.feature.variable_payload
        ):
            raise ValueError(
                f"L2 compression requires a variable-payload connector; "
                f"{self._backend.name} does not support it"
            )
        self._cache_group = partition_id
        if mode != ProfileMode.OFF:
            self._profiler_recorder = ShardedRingRecorder(
                capacity=capacity, output=output
            )

        logger.info(
            "%s is initialized. Using partition_id=%s.", str(self), partition_id
        )

    def __repr__(self) -> str:
        backend_name = "None" if self._backend is None else self._backend.name
        return f"L2Cache(backend={backend_name})"

    def __str__(self) -> str:
        return self.__repr__()

    def __del__(self) -> None:
        self.close()
        logger.info("%s is closed.", str(self))

    async def _ensure_mr_ops_worker(self) -> None:
        if self._mr_ops_queue is None:
            self._loop = asyncio.get_running_loop()
            self._mr_ops_queue = asyncio.Queue()
            self._mr_ops_task = asyncio.create_task(self._mr_ops_worker())

    async def _mr_ops_worker(self) -> None:
        assert self._mr_ops_queue is not None
        while True:
            item = await self._mr_ops_queue.get()
            if item is None:
                self._mr_ops_queue.task_done()
                break

            mr, func, kwargs = item
            try:
                await func(mr, **kwargs)
            except Exception as e:
                logger.warning("MR op failed: %s", e)
            finally:
                self._mr_ops_queue.task_done()

    def _enqueue_mr_op(
        self,
        mr: ConnectorZeroCopyMemoryRegion,
        func: Callable,
        **kwargs,
    ) -> None:
        assert self._loop is not None
        try:
            self._loop.call_soon_threadsafe(
                self._mr_ops_queue.put_nowait,  # type: ignore
                (mr, func, kwargs),
            )
            return
        except Exception:
            pass

    @property
    def feature(self) -> ConnectorFeature:
        return self._backend.feature

    def open(self) -> Status:
        """Open the cache."""
        if not self._use_compact_layout and (
            self._backend.feature.gdr_get or self._backend.feature.gdr_put
        ):
            return Status(
                StatusCodes.INVALID,
                "please set AIBRIX_KV_CACHE_OL_TOKEN_VALIDATION_ENABLED to "
                "false to use GDR",
            )
        return self._backend.open()

    def close(self) -> Status:
        """Close the cache."""
        recorder = getattr(self, "_profiler_recorder", None)
        if recorder is not None:
            recorder.close()
        if self._backend is not None:
            try:
                if self._mr_ops_queue is not None and self._loop is not None:
                    self._loop.call_soon_threadsafe(
                        self._mr_ops_queue.put_nowait, None
                    )  # type: ignore
            except Exception:
                pass
            return self._backend.close()
        return Status.ok()

    def _trace(
        self, context: TraceContext | None, result=None
    ) -> _L2Trace | None:
        if (
            self._profiler_mode == ProfileMode.OFF
            or context is None
            or context.mode == ProfileMode.OFF
        ):
            return None
        try:
            return _L2Trace(self, context, result)
        except Exception:
            self._profiler_recorder.invalidate("observation_error")
            return None

    def register_slabs(self, slabs: List[torch.Tensor]) -> Status:
        if not self._backend.feature.rdma:
            raise NotImplementedError
        status = self._backend.register_slabs(slabs)
        if not status.is_ok():
            return status
        return status

    @nvtx_range("prefetch", "kv_cache_ol.L2Cache")
    async def prefetch(
        self,
        prefix: KVCacheKeyTypes | None,
        query: KVCacheKeyTypes,
    ) -> Status:
        """Prefetch kv tensors from the cache.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
        Returns:
            The status of the prefetch operation.
        """
        if not self._backend.feature.prefetch:
            return Status.ok()

        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        await asyncio.gather(
            *(
                self._prefetch_impl(k)
                for _, k in self._cache_block_keys(prefix, query)
            ),
            return_exceptions=False,  # backend returns exception as status
        )
        return Status.ok()

    async def _prefetch_impl(self, cache_key: str | bytes) -> Status:
        await self._backend.prefetch(cache_key)
        return Status.ok()

    @nvtx_range("exists", "kv_cache_ol.L2Cache")
    @MeasurableBase.measure(MetricRecorder.OP.EXISTS)
    async def exists(
        self,
        prefix: KVCacheKeyTypes | None,
        query: KVCacheKeyTypes,
    ) -> Status[int]:
        """Check if kv tensors exist in the cache.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
        Returns:
            The number of blocks that exist in the cache.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        total = 0
        for key_batch in self._cache_block_key_batches(prefix, query):
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for real_key, key_str in key_batch:
                    tasks.append(tg.create_task(self._backend.exists(key_str)))

            if len(tasks) == 0:
                break

            should_break = False
            for i in range(len(tasks)):
                if not tasks[i].done() or not tasks[i].result().is_ok():
                    should_break = True
                    break
                total += 1

            if should_break:
                break

        if total == 0:
            return Status(StatusCodes.NOT_FOUND)

        return Status.ok(total)

    @nvtx_range("put", "kv_cache_ol.L2Cache")
    @MeasurableBase.measure(MetricRecorder.OP.PUT)
    async def put(
        self,
        prefix: KVCacheKeyTypes | None,
        query: KVCacheKeyTypes,
        kv_tensors: (MemoryRegion | Sequence[MemoryRegion] | KVCacheHandle),
        *,
        trace_context: TraceContext | None = None,
    ) -> Status[int]:
        """Put kv tensors to the cache.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
            kv_tensors: kv tensors or cache handles.
        Returns:
            The status of the put operation and the number of blocks.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        if len(query) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        # If it is not a full block, we don't need to cache it.
        if len(query) // self.block_ntokens == 0:
            return Status.ok(0)

        if self._backend.feature.zero_copy:
            if isinstance(kv_tensors, KVCacheHandle):
                ret = len(kv_tensors)
                # Seal all the MRs
                for mr in kv_tensors.memory_regions:
                    ext_mr = cast(ExternalMemoryRegion, mr)
                    ext_mr._on_release = None
                    connector_mr = ext_mr.depends_on
                    if connector_mr is None:
                        continue
                    self._enqueue_mr_op(connector_mr, self._backend.seal)
                return Status.ok(ret)
            else:
                raise ValueError(
                    "kv_tensors must be KVCacheHandle when zero_copy isenabled."
                )

        if isinstance(kv_tensors, MemoryRegion):
            # `kv_tensors` comes from L1Cache and should be only one block
            assert len(query) // self.block_ntokens == 1, (
                f"len(query)={len(query)}"
            )
            blocks = tuple([kv_tensors])
        elif isinstance(kv_tensors, Sequence):
            assert isinstance(kv_tensors[0], MemoryRegion)
            blocks = tuple(kv_tensors)
        elif isinstance(kv_tensors, KVCacheHandle):
            if len(query) != len(kv_tensors) * self.block_ntokens:
                return Status(
                    StatusCodes.INVALID,
                    (
                        f"Number of query tokens {len(query)} is not equal to "
                        f"the number of tokens in kv tensors "
                        f"{len(kv_tensors) * self.block_ntokens}."
                    ),
                )

            blocks = tuple(kv_tensors.memory_regions)  # type: ignore
        else:
            raise ValueError(f"Unsupported type {type(kv_tensors).__name__}")

        keys = tuple(self._cache_block_keys(prefix, query))
        trace = self._trace(trace_context)
        # use mget if mput_mget is enabled
        if self._compression is None and self._backend.feature.mput_mget:
            block_batches = self._backend.get_batches(
                keys, blocks, self.op_batch
            )
            try:
                return await self._mput_impl(block_batches, trace)
            finally:
                if trace is not None:
                    with trace.observation():
                        trace.finish()
        else:
            block_batches = tuple(batched(zip(keys, blocks), self.op_batch))
            try:
                return await self._put_impl(block_batches, trace)
            finally:
                if trace is not None:
                    with trace.observation():
                        trace.finish()

    async def _mput_impl(
        self,
        block_batches: Sequence[
            Sequence[Tuple[Any, MemoryRegion | Sequence[MemoryRegion]]]
        ],
        trace: _L2Trace | None = None,
    ) -> Status[int]:
        num_processed_blocks = 0
        for batch in block_batches:
            num_blocks_in_batch = len(batch)
            key_pairs, mrs = zip(*batch)
            real_keys, cache_keys = zip(*key_pairs)
            for i, mr in enumerate(mrs):
                if isinstance(mr, ManagedMemoryRegion):
                    if not mr.is_sealed:
                        if not self._use_compact_layout:
                            mr.pack_tokens(
                                prefix=real_keys[i][: -self.block_ntokens],
                                query=real_keys[i][-self.block_ntokens :],
                            )
                        mr.seal()

            started = trace.clock() if trace is not None else 0
            statuses = await self._backend.mput(cache_keys, mrs)
            if trace is not None:
                with trace.observation():
                    trace.storage(
                        EventType.L2_PUT, cache_keys, mrs, statuses, started
                    )

            if isinstance(statuses, Sequence) and all(
                status.is_ok() for status in statuses
            ):
                # all success, continue to the next batch
                num_processed_blocks += num_blocks_in_batch
                continue
            elif num_processed_blocks > 0:
                # current batch is not the first one.
                # at least one batch is done successfully, return success.
                return Status.ok(num_processed_blocks)
            else:
                # this is the first batch and at least one block in
                # current batch is failed, return error.
                if isinstance(statuses, Status):
                    log_every_n_seconds(
                        logger,
                        logging.ERROR,
                        f"mput failed: {statuses}",
                        n_seconds=3,
                    )
                    return statuses

                failures = [status for status in statuses if not status.is_ok()]
                if len(failures) > 0:
                    return failures[0]
                return Status(StatusCodes.ERROR)

        return Status.ok(num_processed_blocks)

    async def _put_impl(
        self,
        block_batches: Sequence[
            Sequence[Tuple[Any, MemoryRegion | Sequence[MemoryRegion]]]
        ],
        trace: _L2Trace | None = None,
    ) -> Status[int]:
        num_processed_blocks = 0
        for batch in block_batches:
            tasks = []
            num_blocks_in_batch = len(batch)
            async with asyncio.TaskGroup() as tg:
                for key_pair, block in batch:
                    tasks.append(
                        tg.create_task(
                            self._backend_put_impl(key_pair, block, trace)
                        )
                    )

            if len(tasks) == 0:
                return Status(StatusCodes.ERROR)
            elif all(task.done() and task.result().is_ok() for task in tasks):
                # all success, continue to the next batch
                num_processed_blocks += num_blocks_in_batch
                continue
            elif num_processed_blocks > 0:
                # current batch is not the first one.
                # at least one batch is done successfully, return success.
                return Status.ok(num_processed_blocks)
            else:
                # this is the first batch and at least one block in
                # current batch is failed, return error.
                failures = [
                    task
                    for task in tasks
                    if task.done() and not task.result().is_ok()
                ]
                if len(failures) > 0:
                    return failures[0].result()
                return Status(StatusCodes.ERROR)

        return Status.ok(num_processed_blocks)

    async def _backend_put_impl(
        self,
        key_pair: Tuple[KVCacheKeyTypes, bytes],
        mr: MemoryRegion | Sequence[MemoryRegion],
        trace: _L2Trace | None = None,
    ) -> Status:
        """Put kv tensors to the backend.
        Args:
            key_pair: I.e., real_key and cache_key.
                real_key (KVCacheKeyTypes): The real key of the kv tensors.
                cache_key (bytes): The cache key of the kv tensors.
            mr: Memory regions of kv tensors.
        Returns:
            The status of the put operation.
        """
        real_key, cache_key = key_pair
        if isinstance(mr, ManagedMemoryRegion) and not mr.is_sealed:
            if not self._use_compact_layout:
                mr.pack_tokens(
                    prefix=real_key[: -self.block_ntokens],
                    query=real_key[-self.block_ntokens :],
                )
            mr.seal()
        if self._compression is None:
            started = trace.clock() if trace is not None else 0
            status = await self._backend.put(cache_key, mr)
            if trace is not None:
                with trace.observation():
                    trace.storage(
                        EventType.L2_PUT, [cache_key], [mr], status, started
                    )
            return status
        if not isinstance(mr, MemoryRegion):
            return Status(
                StatusCodes.INVALID,
                "Compressed L2 storage requires one contiguous MemoryRegion",
            )
        started = trace.clock() if trace is not None else 0
        cpu_started = time.process_time_ns()
        dependency = ()
        try:
            block = self._compression.encode(
                mr.tobytes(),
                shape=self.block_shape,
                dtype=str(self.block_dtype),
                layout=self.block_layout.name,
                block_key=cache_key.hex(),
                generation=0,
                cache_group=self._cache_group,
            )
            encoded = self._compression.serialize(block)
            if trace is not None:
                with trace.observation():
                    # Provenance: encoded was serialized from this block above.
                    sizes = self._compression.encoded_sizes(
                        block, encoded, trusted=True
                    )
                    dependency = (
                        trace.event(
                            EventType.CODEC_ENCODE,
                            cache_key,
                            started,
                            trace.context.dependency_ids,
                            raw_bytes=block.metadata.raw_length,
                            payload_bytes=sizes.payload,
                            metadata_bytes=sizes.metadata,
                            framing_bytes=sizes.framing,
                            envelope_bytes=sizes.envelope,
                            checksum_algorithm="sha256",
                            checksum_before=block.metadata.checksum,
                            cpu_process_start_ns=cpu_started,
                            cpu_process_end_ns=time.process_time_ns(),
                            result="encoded",
                            **trace.resources(),
                        ),
                    )
                    started = trace.now()
            status = await self._backend.put_bytes(cache_key, encoded)
            if trace is not None:
                with trace.observation():
                    operation = trace.event(
                        EventType.L2_PUT,
                        cache_key,
                        started,
                        dependency,
                        destination_tier="L2",
                        envelope_bytes=len(encoded),
                        connector_write_bytes=len(encoded)
                        if status.is_ok()
                        else None,
                        result="stored" if status.is_ok() else "store_error",
                    )
                    trace.failed_status(
                        status, cache_key, started, (operation,)
                    )
            return status
        except Exception as exc:
            if trace is not None:
                with trace.observation():
                    trace.event(
                        EventType.ERROR_RECOVERY,
                        cache_key,
                        started,
                        dependency,
                        result="rejected",
                        error_class=type(exc).__name__,
                        restore_result="rejected",
                    )
            return Status(StatusCodes.ERROR, value=exc)

    @nvtx_range("get", "kv_cache_ol.L2Cache")
    @MeasurableBase.measure(MetricRecorder.OP.GET)
    async def get(
        self,
        prefix: KVCacheKeyTypes | None,
        query: KVCacheKeyTypes,
        mrs: Sequence[MemoryRegion | Sequence[MemoryRegion]],
        *,
        trace_context: TraceContext | None = None,
        trace_result: list[str] | None = None,
    ) -> Status[int]:
        """Get kv tensors from the cache.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
            mrs: Memory regions to place the fetched kv tensors.
        Returns:
            The number of blocks that are fetched.
        """
        assert mrs is not None
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        assert len(mrs) == len(query) // self.block_ntokens

        keys = tuple(self._cache_block_keys(prefix, query))
        trace = self._trace(trace_context, trace_result)
        # use mput if mput_mget is enabled
        if self._compression is None and self._backend.feature.mput_mget:
            block_batches = tuple(
                self._backend.get_batches(keys, mrs, self.op_batch)
            )
            try:
                return await self._mget_impl(block_batches, trace)
            finally:
                if trace is not None:
                    with trace.observation():
                        trace.finish()
        else:
            block_batches = tuple(batched(zip(keys, mrs), self.op_batch))
            try:
                return await self._get_impl(block_batches, trace)
            finally:
                if trace is not None:
                    with trace.observation():
                        trace.finish()

    async def _mget_impl(
        self,
        block_batches: Sequence[
            Sequence[Tuple[Any, MemoryRegion | Sequence[MemoryRegion]]]
        ],
        trace: _L2Trace | None = None,
    ) -> Status[int]:
        nr = 0
        for batch in block_batches:
            status = await self._backend_mget_impl(*zip(*batch), trace=trace)

            should_break = False
            if not status.is_ok():
                should_break = True
                break
            nr += status.get()

            if should_break:
                break

        if nr == 0:
            return Status(StatusCodes.NOT_FOUND)

        return Status.ok(nr)

    async def _backend_mget_impl(
        self,
        key_pairs: Sequence[Tuple[KVCacheKeyTypes, bytes]],
        mrs: Sequence[MemoryRegion | Sequence[MemoryRegion]],
        trace: _L2Trace | None = None,
    ) -> Status[int]:
        """Get kv tensors from the backend using mget.
        Args:
            key_pairs: I.e., a sequence of real_key and cache_key pairs.
                real_key (KVCacheKeyTypes): The real key of the kv tensors.
                cache_key (bytes): The cache key of the kv tensors.
            mrs: Memory regions to place the fetched kv tensors.
        Returns:
            Status of the mget operation.
            Number of blocks that are fetched.
        """
        real_keys, cache_keys = zip(*key_pairs)
        started = trace.clock() if trace is not None else 0
        statuses = await self._backend.mget(cache_keys, mrs)
        if trace is not None:
            with trace.observation():
                trace.storage(
                    EventType.L2_GET, cache_keys, mrs, statuses, started
                )
        if isinstance(statuses, Status):
            status = cast(Status, statuses)
            if not status.is_ok():
                log_every_n_seconds(
                    logger,
                    logging.ERROR,
                    f"mget failed: {status}",
                    n_seconds=3,
                )
                return status
            statuses = [status] * len(mrs)

        is_managed_mr = isinstance(mrs[0], ManagedMemoryRegion)
        nr: int = 0
        for i, status in enumerate(statuses):
            if not status.is_ok():
                continue
            # Bypass token validation if MR is using compact layout
            if is_managed_mr and not self._use_compact_layout:
                mr = mrs[i]
                mr.block_nbytes = self.block_nbytes  # type: ignore
                prefix_in_mr, query_in_mr = mr.unpack_tokens()  # type: ignore
                if not self._tokens_match(
                    real_keys[i], prefix_in_mr, query_in_mr
                ):
                    continue
            nr += 1

        if nr == 0:
            return Status(StatusCodes.NOT_FOUND)
        return Status.ok(nr)

    async def _get_impl(
        self,
        block_batches: Sequence[
            Sequence[Tuple[Any, MemoryRegion | Sequence[MemoryRegion]]]
        ],
        trace: _L2Trace | None = None,
    ) -> Status[int]:
        nr = 0
        first_failure: Status | None = None
        for batch in block_batches:
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for key_pair, mr in batch:
                    tasks.append(
                        tg.create_task(
                            self._backend_get_impl(key_pair, mr, trace)
                        )
                    )

            if len(tasks) == 0:
                break

            should_break = False
            for i in range(len(tasks)):
                if not tasks[i].done() or not tasks[i].result().is_ok():
                    if tasks[i].done():
                        first_failure = tasks[i].result()
                    should_break = True
                    break
                nr += 1

            if should_break:
                break

        if nr == 0:
            if first_failure is not None and not first_failure.is_not_found():
                return first_failure
            return Status(StatusCodes.NOT_FOUND)

        return Status.ok(nr)

    @nvtx_range("flush", "kv_cache_ol.L2Cache")
    def flush(self) -> Status:
        """Synchronously wait until all queued MR ops are processed."""
        if self._mr_ops_queue is None:
            return Status.ok()
        try:
            assert self._loop is not None
            fut = asyncio.run_coroutine_threadsafe(
                self._mr_ops_queue.join(), self._loop
            )
            fut.result(timeout=60)
        except TimeoutError:
            # timed out
            return Status(StatusCodes.TIMEOUT)
        except Exception as e:
            return Status(StatusCodes.ERROR, value=e)
        return Status.ok()

    async def _backend_get_impl(
        self,
        key_pair: Tuple[KVCacheKeyTypes, bytes],
        mr: MemoryRegion | Sequence[MemoryRegion],
        trace: _L2Trace | None = None,
    ) -> Status:
        """Get kv tensors from the backend.
        Args:
            key_pair: I.e., real_key and cache_key.
                real_key (KVCacheKeyTypes): The real key of the kv tensors.
                cache_key (bytes): The cache key of the kv tensors.
            mr: Memory regions to place the fetched kv tensors.
        Returns:
            The status of the get operation.
        """
        real_key, cache_key = key_pair
        if self._compression is None:
            started = trace.clock() if trace is not None else 0
            status = await self._backend.get(cache_key, mr)
            if trace is not None:
                with trace.observation():
                    trace.storage(
                        EventType.L2_GET, [cache_key], [mr], status, started
                    )
            if not status.is_ok():
                return status
        else:
            if not isinstance(mr, MemoryRegion):
                return Status(
                    StatusCodes.INVALID,
                    "Compressed L2 storage requires one contiguous "
                    "MemoryRegion",
                )
            started = trace.clock() if trace is not None else 0
            dependency = ()
            try:
                status = await self._backend.get_bytes(cache_key)
            except Exception as exc:
                if trace is not None:
                    with trace.observation():
                        trace.event(
                            EventType.ERROR_RECOVERY,
                            cache_key,
                            started,
                            trace.context.dependency_ids,
                            result="rejected",
                            restore_result="rejected",
                            error_class=type(exc).__name__,
                        )
                raise
            if trace is not None:
                with trace.observation():
                    dependency = (
                        trace.event(
                            EventType.L2_GET,
                            cache_key,
                            started,
                            trace.context.dependency_ids,
                            source_tier="L2",
                            connector_read_bytes=(
                                len(status.get()) if status.is_ok() else None
                            ),
                            result=(
                                "loaded"
                                if status.is_ok()
                                else "not_found"
                                if status.is_not_found()
                                else "load_error"
                            ),
                        ),
                    )
                    trace.failed_status(status, cache_key, started, dependency)
            if not status.is_ok():
                return status
            try:
                envelope = status.get()
                started = trace.clock() if trace is not None else 0
                cpu_started = time.process_time_ns()
                block = self._compression.deserialize(envelope)
                raw = self._compression.decode(
                    block,
                    expected_block_key=cache_key.hex(),
                    expected_generation=0,
                    expected_cache_group=self._cache_group,
                    expected_shape=self.block_shape,
                    expected_dtype=str(self.block_dtype),
                    expected_layout=self.block_layout.name,
                )
                if len(raw) != mr.length:
                    raise ValueError(
                        f"restored block length {len(raw)} != "
                        f"MR length {mr.length}"
                    )
                if trace is not None:
                    with trace.observation():
                        # Provenance: block came from this exact envelope.
                        sizes = self._compression.encoded_sizes(
                            block, envelope, trusted=True
                        )
                        dependency = (
                            trace.event(
                                EventType.CODEC_DECODE,
                                cache_key,
                                started,
                                dependency,
                                raw_bytes=len(raw),
                                payload_bytes=sizes.payload,
                                metadata_bytes=sizes.metadata,
                                framing_bytes=sizes.framing,
                                envelope_bytes=sizes.envelope,
                                result="decoded",
                                checksum_algorithm="sha256",
                                checksum_after=block.metadata.checksum,
                                cpu_process_start_ns=cpu_started,
                                cpu_process_end_ns=time.process_time_ns(),
                                **trace.resources(),
                            ),
                        )
                        started = trace.now()
                mr.fill(raw)
                if trace is not None:
                    with trace.observation():
                        trace.event(
                            EventType.KV_IMPORT,
                            cache_key,
                            started,
                            dependency,
                            phase="restore",
                            raw_bytes=len(raw),
                            result="restored",
                            restore_result="restored",
                        )
            except Exception as exc:
                if trace is not None:
                    with trace.observation():
                        trace.event(
                            EventType.ERROR_RECOVERY,
                            cache_key,
                            started,
                            dependency,
                            result="rejected",
                            restore_result="rejected",
                            error_class=type(exc).__name__,
                        )
                return Status(StatusCodes.ERROR, value=exc)

        # Bypass token validation if MR is using compact layout
        if isinstance(mr, ManagedMemoryRegion) and not self._use_compact_layout:
            # check if tokens match
            mr.block_nbytes = self.block_nbytes
            prefix_in_mr, query_in_mr = mr.unpack_tokens()
            if self._tokens_match(real_key, prefix_in_mr, query_in_mr):
                return Status.ok()
            else:
                return Status(StatusCodes.NOT_FOUND, "tokens mismatch")

        return Status.ok()

    def _tokens_match(
        self,
        real_key: KVCacheKeyTypes,
        prefix_in_mr: KVCacheKeyTypes | None,
        query_in_mr: KVCacheKeyTypes | None,
    ) -> bool:
        """Check if the tokens in mr match the real key.
        Args:
            real_key (KVCacheKeyTypes): The real key of the kv tensors.
            prefix_in_mr (KVCacheKeyTypes | None): The prefix tokens in mr.
            query_in_mr (KVCacheKeyTypes | None): The query tokens in mr.
        Returns:
            True if the tokens in mr match the real key, False otherwise.
        """
        try:
            if query_in_mr is None or len(query_in_mr) != self.block_ntokens:
                return False
            if prefix_in_mr is not None:
                all_tokens = prefix_in_mr + query_in_mr
            else:
                all_tokens = query_in_mr
            is_identical = all_tokens == real_key
            if is_identical:
                return True
            else:
                return False
        except Exception:
            return False

    @nvtx_range("allocate", "kv_cache_ol.L2Cache")
    @MeasurableBase.measure(MetricRecorder.OP.ALLOCATE)
    async def allocate(
        self,
        prefix: KVCacheKeyTypes | None,
        query: KVCacheKeyTypes,
        sizes: Sequence[int],
    ) -> Status[Sequence[MemoryRegion]]:
        """Allocate pinned memory regions to store the kv tensors.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
            sizes: The sizes of the memory regions.
        Returns:
            The memory regions to store the kv tensors.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        if len(query) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        # If it is not a full block, we don't need to cache it.
        if len(query) // self.block_ntokens == 0 or len(sizes) == 0:
            return Status(StatusCodes.INVALID)

        await self._ensure_mr_ops_worker()

        mrs: List[MemoryRegion] = []
        si = 0
        for key_batch in self._cache_block_key_batches(prefix, query):
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for _, key_str in key_batch:
                    if si >= len(sizes):
                        break
                    tasks.append(
                        tg.create_task(
                            self._backend.allocate(key_str, sizes[si])
                        )
                    )
                    si += 1

            if len(tasks) == 0:
                break

            i = 0
            for i in range(len(tasks)):
                if not tasks[i].done() or not tasks[i].result().is_ok():
                    break
                connector_mr: ConnectorZeroCopyMemoryRegion = (
                    tasks[i].result().get()
                )
                slab = buffer_to_tensor(connector_mr.addr, connector_mr.length)

                def on_release(
                    x: torch.Tensor, y: int, z: int, _mr=connector_mr
                ) -> None:
                    # We drop the MR by default. Only seal it if it is put into
                    # the cache. Therefore, the actual seal op is done in the
                    # put operation.
                    self._enqueue_mr_op(_mr, self._backend.drop)

                mr = ExternalMemoryRegion(
                    slab,
                    0,
                    connector_mr.length,
                    on_release,
                    depends_on=connector_mr,
                )
                mrs.append(mr)

            if i + 1 < len(tasks):
                # release any mrs that are not returned
                for j in range(i, len(tasks)):
                    if not tasks[j].done() or not tasks[j].result().is_ok():
                        continue
                    mr_to_drop = tasks[j].result().get()
                    self._enqueue_mr_op(mr_to_drop, self._backend.drop)
                break

        return Status.ok(mrs)

    @nvtx_range("acquire", "kv_cache_ol.L2Cache")
    @MeasurableBase.measure(MetricRecorder.OP.ACQUIRE)
    async def acquire(
        self,
        prefix: KVCacheKeyTypes | None,
        query: KVCacheKeyTypes,
        *,
        trace_context: TraceContext | None = None,
        trace_result: list[str] | None = None,
    ) -> Status[Sequence[MemoryRegion]]:
        """Acquire pinned memory regions that store the kv tensors.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
        Returns:
            The memory regions that store the kv tensors.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        await self._ensure_mr_ops_worker()

        trace = self._trace(trace_context, trace_result)
        mrs: List[MemoryRegion] = []
        for key_batch in self._cache_block_key_batches(prefix, query):
            start = trace.clock() if trace else 0
            trace_keys = [] if trace else None
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for _, key_str in key_batch:
                    if trace_keys is not None:
                        trace_keys.append(key_str)
                    tasks.append(tg.create_task(self._backend.acquire(key_str)))

            if trace:
                with trace.observation():
                    for key, task in zip(trace_keys, tasks):
                        status = task.result()
                        trace.event(
                            EventType.L2_GET,
                            key,
                            start,
                            phase="zero_copy_acquire",
                            source_tier="L2",
                            result="loaded"
                            if status.is_ok()
                            else "not_found"
                            if status.is_not_found()
                            else "storage_error",
                        )

            if len(tasks) == 0:
                break

            i = 0
            for i in range(len(tasks)):
                if not tasks[i].done() or not tasks[i].result().is_ok():
                    break
                connector_mr: ConnectorZeroCopyMemoryRegion = (
                    tasks[i].result().get()
                )
                slab = buffer_to_tensor(connector_mr.addr, connector_mr.length)

                def on_release(
                    x: torch.Tensor,
                    y: int,
                    z: int,
                    _mr=connector_mr,
                ) -> None:
                    self._enqueue_mr_op(_mr, self._backend.release)

                mr = ExternalMemoryRegion(
                    slab,
                    0,
                    connector_mr.length,
                    on_release=on_release,
                    depends_on=connector_mr,
                )
                mrs.append(mr)

            if i + 1 < len(tasks):
                # release any mrs that are not returned
                for j in range(i, len(tasks)):
                    if not tasks[j].done() or not tasks[j].result().is_ok():
                        continue
                    mr_to_release = tasks[j].result().get()
                    self._enqueue_mr_op(
                        mr_to_release,
                        self._backend.release,
                    )
                break

        if trace:
            with trace.observation():
                trace.finish()
        if len(mrs) == 0:
            return Status(StatusCodes.NOT_FOUND)

        return Status.ok(mrs)

    @nvtx_range("delete", "kv_cache_ol.L2Cache")
    async def delete(
        self, prefix: KVCacheKeyTypes | None, query: KVCacheKeyTypes
    ) -> Status:
        """Delete kv tensors from the cache.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
        Returns:
            The status of the delete operation.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        for _, key_str in self._cache_block_keys(prefix, query):
            await self._backend.delete(key_str)
        return Status.ok()

    def _cache_block_keys(
        self, prefix: KVCacheKeyTypes | None, query: KVCacheKeyTypes
    ) -> Iterator[Tuple[KVCacheKeyTypes, bytes]]:
        """Get the cache block keys of the kv tensors.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
        Returns:
            The cache block keys of the kv tensors.
        """
        if isinstance(query, BlockHashes):
            for i in range(0, len(query), self.block_ntokens):
                yield tuple(
                    (
                        query[i : i + self.block_ntokens],
                        "".join(query[i : i + self.block_ntokens]).encode(
                            "utf-8"
                        ),
                    )
                )
        else:
            if prefix is not None:
                assert isinstance(prefix, TokenListView)
            assert isinstance(query, TokenListView)
            yield from iter(self.key_builder.build(prefix, query))

    def _cache_block_key_batches(
        self, prefix: KVCacheKeyTypes | None, query: KVCacheKeyTypes
    ) -> Iterator[Iterator[Tuple[KVCacheKeyTypes, bytes]]]:
        """Get the cache block key batchs.
        Args:
            prefix: The prefix tokens of the kv tensors.
            query: The query tokens of the kv tensors.
        Returns:
            The cache block key batchs of the kv tensors.
        """
        for batch in batched(
            self._cache_block_keys(prefix, query), self.op_batch
        ):
            yield iter(batch)
