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
"""Scheduler-to-worker jobs and worker-to-scheduler results."""

from dataclasses import dataclass, field
from typing import Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

from ....compression_profiler import TraceContext
from .logic import Part


@dataclass
class LoadJob:
    req_id: str
    block_hash: bytes
    boundary: int
    # The record's state and spans, each one stored object.
    parts: list[Part]
    context: Optional[TraceContext] = None
    # Profiler operations this job waited for (a restore: its eviction).
    after: tuple[str, ...] = ()


@dataclass
class SaveJob(LoadJob):
    pass


@dataclass
class HybridMetadata(KVConnectorMetadata):
    loads: list[LoadJob] = field(default_factory=list)
    saves: list[SaveJob] = field(default_factory=list)
    # Diagnostic only: compare a record with vLLM's own local-hit blocks.
    verifies: list[LoadJob] = field(default_factory=list)


@dataclass
class HybridWorkerMetadata(KVConnectorWorkerMetadata):
    # (part, block hash, boundary, start, rank, L2_PUT operation id or "").
    stored: list[tuple[str, bytes, int, int, int, str]] = field(
        default_factory=list
    )
    # (part, block hash, boundary, start) of parts that failed to load/save.
    failed: list[tuple[str, bytes, int, int]] = field(default_factory=list)
    save_failed: list[tuple[str, bytes, int, int]] = field(default_factory=list)

    def aggregate(self, other: "KVConnectorWorkerMetadata"):
        assert isinstance(other, HybridWorkerMetadata)
        return HybridWorkerMetadata(
            self.stored + other.stored,
            self.failed + other.failed,
            self.save_failed + other.save_failed,
        )
