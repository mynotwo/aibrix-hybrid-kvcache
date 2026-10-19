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

"""Per-operation tracing shared by every component that stores KV data."""

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from .. import envs
from .context import TraceContext
from .recorder import NullRecorder, ShardedRingRecorder
from .schema import EventType, ProfileMode, TraceEvent


def profiler_config():
    mode = envs.AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_MODE
    if mode not in ("OFF", "LIGHT", "DETAILED"):
        raise ValueError(
            "AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_MODE must be "
            "OFF, LIGHT or DETAILED"
        )
    capacity = envs.AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_CAPACITY
    if not capacity.isascii() or not capacity.isdecimal() or int(capacity) <= 0:
        raise ValueError(
            "AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_CAPACITY must be "
            "a positive integer"
        )
    output = envs.AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_OUTPUT
    if mode != "OFF" and not output:
        raise ValueError(
            "AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_OUTPUT "
            "must be a JSONL path"
        )
    request_ids = envs.AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_REQUEST_IDS
    requests = frozenset(
        request.strip()
        for request in request_ids.split(",")
        if request.strip()
    )
    return ProfileMode(mode.lower()), int(capacity), output, requests


def open_recorder(role: str):
    """(recorder, mode, requests) of this process.

    Each process of a multi-process server writes ``<dir>/<role>/<name>``,
    so recorders never share an output, owner claim or run status.
    """
    mode, capacity, output, requests = profiler_config()
    if mode == ProfileMode.OFF:
        return NullRecorder(), mode, requests
    path = Path(output)
    recorder = ShardedRingRecorder(
        capacity=capacity, output=path.parent / role / path.name
    )
    return recorder, mode, requests


def request_context(request_id: str) -> TraceContext | None:
    """Root context of one request, or None when the run is not named."""
    mode = ProfileMode(
        envs.AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_MODE.lower()
    )
    experiment_id = envs.AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_EXPERIMENT_ID
    run_id = envs.AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_RUN_ID
    if mode == ProfileMode.OFF or not experiment_id or not run_id:
        return None
    return TraceContext(
        experiment_id, run_id, request_id, mode, uuid.uuid4().hex
    )


def request_event(recorder, context: TraceContext, start: int, **values):
    """Record a request's root event; its operations are its children."""
    try:
        recorder.record(
            lambda: TraceEvent(
                experiment_id=context.experiment_id,
                run_id=context.run_id,
                request_id=context.request_id,
                operation_id=context.operation_id,
                parent_operation_id=context.parent_operation_id,
                dependency_ids=context.dependency_ids,
                event_type=EventType.REQUEST,
                start_ns=start,
                end_ns=OperationTrace.now(),
                **values,
            )
        )
    except Exception:
        recorder.invalidate("observation_error")


@dataclass(frozen=True)
class TraceSink:
    """What an operation trace needs from the component that owns it."""

    recorder: NullRecorder | ShardedRingRecorder
    mode: ProfileMode
    requests: frozenset[str]
    codec: str | None
    cache_group: str
    cache_spec: str


class OperationTrace:
    """One call's stage aggregates, or selected request's per-block spans."""

    def __init__(self, sink: TraceSink, context: TraceContext, result=None):
        self.sink = sink
        self.context = context
        self.result = result
        self.rejected_operations: set[str] = set()
        self.detailed = (
            sink.mode == ProfileMode.DETAILED
            and context.mode == ProfileMode.DETAILED
            and context.request_id in sink.requests
        )
        self.aggregates: dict[EventType, dict] = {}
        self.errors: list[dict] = []

    @staticmethod
    def now() -> int:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    @contextmanager
    def observation(self):
        """Only enclose profiler work, never codec, storage or MR writes."""
        try:
            yield
        except Exception:
            self.sink.recorder.invalidate("observation_error")

    def clock(self) -> int:
        with self.observation():
            return self.now()
        return 0

    def resources(self) -> dict[str, int | None]:
        """Low-frequency detailed snapshots; unavailable values stay null."""
        if not self.detailed:
            return {}
        rss_bytes = None
        try:
            with open("/proc/self/statm", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            rss_bytes = resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
        hbm_allocated = hbm_reserved = None
        if torch.cuda.is_available():
            hbm_allocated = torch.cuda.memory_allocated()
            hbm_reserved = torch.cuda.memory_reserved()
        return {
            "rss_bytes": rss_bytes,
            "pinned_bytes": None,
            "hbm_allocated_bytes": hbm_allocated,
            "hbm_reserved_bytes": hbm_reserved,
        }

    def event(
        self, kind, key, start, dependencies=(), *, end=None, **values
    ) -> str:
        """Record ``[start, end)``; ``end`` defaults to now."""
        end = self.now() if end is None else end
        # Errors remain individually observable even in LIGHT mode.
        aggregate = not self.detailed and kind != EventType.ERROR_RECOVERY
        if aggregate and kind in self.aggregates:
            record = self.aggregates[kind]
            record["start_ns"] = min(record["start_ns"], start)
            record["end_ns"] = max(record["end_ns"], end)
            for name, value in values.items():
                if name.endswith("_bytes"):
                    previous = record.get(name)
                    record[name] = (
                        previous + value
                        if previous is not None and value is not None
                        else None
                    )
                elif name in ("result", "restore_result"):
                    if record.get(name) != value:
                        record[name] = "mixed"
            return record["operation_id"]
        operation = self.context.child(
            uuid.uuid4().hex, dependencies=dependencies
        )
        record = dict(
            experiment_id=operation.experiment_id,
            run_id=operation.run_id,
            request_id=operation.request_id,
            operation_id=operation.operation_id,
            parent_operation_id=operation.parent_operation_id,
            dependency_ids=operation.dependency_ids,
            event_type=kind,
            start_ns=start,
            end_ns=end,
            codec=self.sink.codec,
            **values,
        )
        if self.detailed:
            record.update(
                block_key=key.hex(),
                generation=None,
                cache_group=self.sink.cache_group,
                cache_spec=self.sink.cache_spec,
            )
        if aggregate:
            # A multi-block aggregate has no single checksum or block identity.
            for name in ("checksum_before", "checksum_after"):
                record.pop(name, None)
            self.aggregates[kind] = record
        elif not self.detailed:
            # LIGHT errors depend on aggregates accepted only at finish().
            self.errors.append(record)
        else:
            self._record(record)
        return operation.operation_id

    def _record(self, record):
        accepted = False
        try:
            if any(
                dep in self.rejected_operations
                for dep in record["dependency_ids"]
            ):
                return
            accepted = self.sink.recorder.record(
                lambda: TraceEvent(**record)
            )
            if accepted and self.result is not None:
                self.result.append(record["operation_id"])
            elif not accepted:
                self.sink.recorder.invalidate("observation_error")
        finally:
            if not accepted:
                self.rejected_operations.add(record["operation_id"])

    def finish(self):
        for record in self.aggregates.values():
            self._record(record)
        for record in self.errors:
            self._record(record)
