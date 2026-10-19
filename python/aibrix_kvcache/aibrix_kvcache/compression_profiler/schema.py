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

from dataclasses import dataclass, field, fields
from enum import StrEnum
from math import isfinite


class ProfileMode(StrEnum):
    OFF = "off"
    LIGHT = "light"
    DETAILED = "detailed"


class EventType(StrEnum):
    REQUEST = "request"
    PREFILL = "prefill"
    DECODE = "decode"
    KV_EXPORT = "kv_export"
    CODEC_ENCODE = "codec_encode"
    L2_PUT = "l2_put"
    L2_GET = "l2_get"
    CODEC_DECODE = "codec_decode"
    KV_IMPORT = "kv_import"
    FIRST_TOKEN = "first_token"
    OUTPUT_TOKEN = "output_token"
    EVICTION = "eviction"
    FALLBACK = "fallback"
    ERROR_RECOVERY = "error_recovery"


@dataclass(frozen=True, slots=True)
class BlockIdentity:
    """Distinguish versions and cache groups of the same block key."""

    block_key: str
    generation: int
    cache_group: str
    layer: int | None = None
    cache_spec: str = ""


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Scalar metadata only; unavailable observations remain null in JSON."""

    experiment_id: str
    run_id: str
    request_id: str
    operation_id: str
    event_type: EventType
    schema_version: int = field(default=1, init=False)
    parent_operation_id: str = ""
    dependency_ids: tuple[str, ...] = ()
    block_key: str | None = None
    generation: int | None = None
    layer: int | None = None
    cache_group: str | None = None
    layer_observed: bool = False
    cache_spec: str | None = None
    source_tier: str | None = None
    destination_tier: str | None = None
    phase: str = ""
    codec: str | None = None
    result: str = ""
    error_class: str = ""
    restore_result: str = ""
    enqueue_ns: int | None = None
    start_ns: int | None = None
    device_start: float | None = None
    device_end: float | None = None
    # Whole-process CPU clock (every thread), not this operation's CPU cost.
    cpu_process_start_ns: int | None = None
    cpu_process_end_ns: int | None = None
    end_ns: int | None = None
    raw_bytes: int | None = None
    payload_bytes: int | None = None
    metadata_bytes: int | None = None
    framing_bytes: int | None = None
    envelope_bytes: int | None = None
    connector_read_bytes: int | None = None
    connector_write_bytes: int | None = None
    transferred_bytes: int | None = None
    stored_object_bytes: int | None = None
    retrieved_object_bytes: int | None = None
    decode_output_bytes: int | None = None
    workspace_bytes: int | None = None
    rss_bytes: int | None = None
    pinned_bytes: int | None = None
    hbm_allocated_bytes: int | None = None
    hbm_reserved_bytes: int | None = None
    checksum_algorithm: str | None = None
    checksum_before: str | None = None
    checksum_after: str | None = None
    token_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType(self.event_type))
        dependencies = tuple(self.dependency_ids)
        if any(type(value) is not str for value in dependencies):
            raise TypeError("dependency_ids must contain only strings")
        object.__setattr__(self, "dependency_ids", dependencies)
        for item in fields(self):
            if item.name in ("event_type", "dependency_ids"):
                continue
            value = getattr(self, item.name)
            if type(value) not in (str, int, float, bool, type(None)):
                raise TypeError(f"{item.name} must be JSON scalar metadata")
            if type(value) is float and not isfinite(value):
                raise ValueError(f"{item.name} must be finite")

    def to_record(
        self,
    ) -> dict[str, str | int | float | bool | None | list[str]]:
        """Build a detached JSON record, without encoding on the hot path."""
        record = {item.name: getattr(self, item.name) for item in fields(self)}
        record["event_type"] = self.event_type.value
        record["dependency_ids"] = list(self.dependency_ids)
        return record
