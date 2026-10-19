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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from math import isfinite

from .recorder import TraceRunStatus
from .resume_stages import resume_stage_reasons
from .schema import EventType


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    reasons: tuple[str, ...]
    unknowns: tuple[str, ...] = ()

# Codecs that encode on the GPU, so that only the encoding crosses PCIe
# (events spell codecs in either case).
GPU_CODECS = ("ANS",)


def validate_trace(
    records: Iterable[Mapping],
    status: TraceRunStatus,
    *,
    require_complete: bool = False,
) -> ValidationReport:
    """Structural checks; analysis additionally requires closed, named spans."""
    reasons = list(status.reasons)

    def invalid(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if not status.valid:
        invalid("run_invalid")
    if status.dropped_events:
        invalid("dropped_event")
    operations: set[str] = set()
    by_id: dict[str, Mapping] = {}
    dependencies: list[tuple[str, str]] = []
    try:
        for record in records:
            if not isinstance(record, Mapping):
                invalid("record_type")
                continue
            if type(record.get("schema_version")) is not int or (
                record.get("schema_version") != 1
            ):
                invalid("schema_version")
            operation = record.get("operation_id")
            if type(operation) is not str or not operation:
                invalid("operation_id")
                operation = ""
            elif operation in operations:
                invalid("duplicate_operation_id")
            else:
                operations.add(operation)
                by_id[operation] = record
            dependency_ids = record.get("dependency_ids", ())
            if not isinstance(dependency_ids, (list, tuple)) or any(
                type(value) is not str for value in dependency_ids
            ):
                invalid("dependency_ids")
                dependency_ids = ()
            parent = record.get("parent_operation_id", "")
            if type(parent) is not str:
                invalid("dependency_ids")
            elif parent:
                dependencies.append((operation, parent))
            dependencies.extend((operation, dep) for dep in dependency_ids)
            # Host monotonic ns and device timestamps are separate clocks.
            for names in (
                ("enqueue_ns", "start_ns", "end_ns"),
                ("device_start", "device_end"),
            ):
                values = []
                for name in names:
                    value = record.get(name)
                    if value is None:
                        continue
                    if type(value) not in (int, float) or (
                        not isfinite(value) or value < 0
                    ):
                        invalid("span_time")
                    else:
                        values.append(value)
                if any(a > b for a, b in zip(values, values[1:])):
                    invalid("span_order")
            if require_complete:
                for name in ("experiment_id", "run_id", "request_id"):
                    if not isinstance(record.get(name), str) or not (
                        record[name].strip()
                    ):
                        invalid("missing_" + name)
                if any(record.get(n) is None for n in ("start_ns", "end_ns")):
                    invalid("unclosed_span")
                if record.get("event_type") not in set(EventType):
                    invalid("event_type")
                if record.get("block_key") is not None:
                    for name in ("block_key", "cache_group", "cache_spec"):
                        if not isinstance(record.get(name), str) or not (
                            record[name].strip()
                        ):
                            invalid("missing_" + name)
                    generation = record.get("generation")
                    if type(generation) is not int or generation < 0:
                        invalid("missing_generation")
                    if record.get("layer_observed") and (
                        type(record.get("layer")) is not int
                        or record["layer"] < 0
                    ):
                        invalid("missing_layer")
        for operation, dependency in dependencies:
            if dependency == operation:
                invalid("self_dependency")
            if dependency not in operations:
                invalid("missing_dependency")
            elif operation in by_id:
                current, prior = by_id[operation], by_id[dependency]
                if any(
                    current.get(n) != prior.get(n)
                    for n in _shared_identity(current, prior)
                ):
                    invalid("cross_identity_dependency")
                if current.get("block_key") and prior.get("block_key"):
                    if _identity(current) != _identity(prior):
                        invalid("block_identity_conflict")
        graph = {operation: set() for operation in operations}
        for operation, dependency in dependencies:
            if operation in graph and dependency in graph:
                graph[operation].add(dependency)
        try:
            tuple(TopologicalSorter(graph).static_order())
        except CycleError:
            invalid("cycle")
    except Exception:
        invalid("malformed_trace")
    return ValidationReport(not reasons, tuple(reasons))


_IDENTITY = (
    "experiment_id",
    "run_id",
    "block_key",
    "generation",
    "cache_group",
    "cache_spec",
    "layer_observed",
    "layer",
)
_SUCCESS = {
    "encoded",
    "decoded",
    "stored",
    "loaded",
    "restored",
    "copied",
    "ok",
    "success",
    "hit",
}
_BYTE_FIELDS = (
    "raw_bytes",
    "payload_bytes",
    "metadata_bytes",
    "framing_bytes",
    "envelope_bytes",
    "connector_read_bytes",
    "connector_write_bytes",
    "workspace_bytes",
    "decode_output_bytes",
    "stored_object_bytes",
    "retrieved_object_bytes",
    "transferred_bytes",
)


def _identity(record):
    return tuple(record.get(name) for name in _IDENTITY)


def _shared_identity(current: Mapping, prior: Mapping) -> tuple[str, ...]:
    # Only a restore may depend on another request: its eviction (local
    # miss) depends on the put of the object an earlier request stored.
    if (current.get("event_type"), prior.get("event_type")) == (
        "eviction",
        "l2_put",
    ):
        return ("run_id", "experiment_id")
    return ("request_id", "run_id", "experiment_id")


def validate_byte_accounting(records: Iterable[Mapping]) -> ValidationReport:
    """Check observed logical object bytes, never physical bus traffic.

    v1 decode.raw_bytes is the measured decode output; restore.raw_bytes
    provides a second observation. Additional object counters are optional.
    Unknown observations are reported, not replaced with zero.
    """
    records = list(records)
    reasons, unknowns = [], []

    def equal(row, name, left, right):
        if left is None or right is None:
            unknowns.append(f"{row.get('operation_id')}:{name}")
        elif left != right:
            reasons.append(name)

    def total(values):
        return None if any(v is None for v in values) else sum(values)

    for row in records:
        if not isinstance(row, Mapping):
            return ValidationReport(False, ("record_type",))
        for name in _BYTE_FIELDS:
            value = row.get(name)
            if value is not None and (type(value) is not int or value < 0):
                reasons.append("byte_value")
    if reasons:
        return ValidationReport(False, tuple(dict.fromkeys(reasons)))
    by_id = {r.get("operation_id"): r for r in records}
    successful = [r for r in records if r.get("result") in _SUCCESS]
    identities: dict[tuple, list[Mapping]] = {}
    for row in successful:
        kind = row.get("event_type")
        if kind in ("codec_encode", "codec_decode"):
            if row.get("raw_bytes") is None:
                equal(row, "decode_output_bytes" if kind == "codec_decode"
                      else "raw_bytes", None, None)
            equal(
                row,
                "envelope_bytes",
                total(
                    [
                        row.get(n)
                        for n in (
                            "payload_bytes",
                            "metadata_bytes",
                            "framing_bytes",
                        )
                    ]
                ),
                row.get("envelope_bytes"),
            )
        if row.get("decode_output_bytes") is not None:
            equal(
                row,
                "decode_output_bytes",
                row.get("raw_bytes"),
                row.get("decode_output_bytes"),
            )
        for direction, object_field in (
            ("write", "stored_object_bytes"),
            ("read", "retrieved_object_bytes"),
        ):
            name = f"connector_{direction}_bytes"
            applicable = (kind, direction) in (("l2_put", "write"),
                                              ("l2_get", "read"))
            if applicable or row.get(name) is not None:
                equal(
                    row,
                    name,
                    row.get(name),
                    row.get(object_field, row.get("envelope_bytes")),
                )
        if row.get("checksum_before") is not None:
            equal(
                row,
                "checksum",
                row.get("checksum_before"),
                row.get("checksum_after"),
            )
        for dep in row.get("dependency_ids", ()):
            prior = by_id.get(dep)
            if prior is None or prior.get("result") not in _SUCCESS:
                continue
            if kind == "codec_decode" and prior.get("event_type") == "l2_get":
                equal(
                    row,
                    "connector_read_bytes",
                    row.get("envelope_bytes"),
                    prior.get("connector_read_bytes"),
                )
            if kind == "l2_put" and prior.get("event_type") == "codec_encode":
                equal(
                    row,
                    "connector_write_bytes",
                    row.get("connector_write_bytes"),
                    prior.get("envelope_bytes"),
                )
            if (
                kind == "kv_import"
                and row.get("phase") == "restore"
                and prior.get("event_type") == "codec_decode"
            ):
                equal(
                    row,
                    "decode_output_bytes",
                    row.get("raw_bytes"),
                    prior.get("raw_bytes"),
                )
        if row.get("block_key") and all(
            row.get(n) is not None
            for n in ("generation", "cache_group", "cache_spec")
        ):
            identities.setdefault(_identity(row), []).append(row)
        children = [
            r
            for r in records
            if r.get("parent_operation_id") == row.get("operation_id")
            and r.get("event_type") == kind
            and r.get("block_key")
        ]
        if row.get("block_key") is None and children:
            if len({_identity(r) for r in children}) != len(children):
                reasons.append("aggregate_duplicate_block")
            for field, aggregate in (
                ("raw_bytes", "raw_bytes"),
                ("envelope_bytes", "envelope_bytes"),
                ("envelope_bytes", "transferred_bytes"),
            ):
                if aggregate == "transferred_bytes" and aggregate not in row:
                    continue
                values = [
                    r.get(field) if r in successful else None for r in children
                ]
                equal(
                    row,
                    "operation_" + aggregate,
                    row.get(aggregate),
                    total(values),
                )
    for group in identities.values():
        raw = {r["raw_bytes"] for r in group if r.get("raw_bytes") is not None}
        if len(raw) > 1:
            reasons.append("decode_output_bytes")
        checksums = {
            r[n]
            for r in group
            for n in ("checksum_before", "checksum_after")
            if r.get(n)
        }
        algorithms = {
            r.get("checksum_algorithm")
            for r in group
            if r.get("checksum_before") or r.get("checksum_after")
        }
        if (
            len(algorithms) == 1
            and None not in algorithms
            and len(checksums) > 1
        ):
            reasons.append("checksum")
    return ValidationReport(
        not reasons,
        tuple(dict.fromkeys(reasons)),
        tuple(dict.fromkeys(unknowns)),
    )


_QWEN_RESUME_CHAIN = (
    "kv_export",
    "codec_encode",
    "l2_put",
    "eviction",
    "l2_get",
    "codec_decode",
    "kv_import",
)
_RESOURCE_FIELDS = (
    "rss_bytes",
    "pinned_bytes",
    "hbm_allocated_bytes",
    "hbm_reserved_bytes",
)


def has_resume_path(
    records: list[Mapping], terminals: Iterable[str] | None = None
) -> bool:
    """A kv_import whose dependency ancestry holds the chain in order.

    ``terminals`` limits the kv_import operations considered. Order comes
    from dependency edges, never from file position: shards and background
    threads export events out of causal order.
    """
    by_id = {record.get("operation_id"): record for record in records}
    wanted = _QWEN_RESUME_CHAIN[::-1]
    seen: set[tuple[str, int]] = set()
    allowed = None if terminals is None else set(terminals)
    stack = [
        (record.get("operation_id"), 0)
        for record in records
        if record.get("event_type") == wanted[0]
        and (allowed is None or record.get("operation_id") in allowed)
    ]
    while stack:
        operation, index = stack.pop()
        record = by_id.get(operation)
        if record is None or (operation, index) in seen:
            continue
        seen.add((operation, index))
        if record.get("event_type") == wanted[index]:
            index += 1
            if index == len(wanted):
                return True
        stack.extend(
            (dependency, index)
            for dependency in record.get("dependency_ids") or ()
        )
    return False


def validate_qwen_resume_trace(
    records: Iterable[Mapping], status: TraceRunStatus
) -> ValidationReport:
    """Validate one real external reload without conflating byte domains."""
    records = list(records)
    reports = (
        validate_trace(records, status, require_complete=True),
        validate_byte_accounting(records),
    )
    reasons = [reason for report in reports for reason in report.reasons]
    unknowns = [unknown for report in reports for unknown in report.unknowns]
    kinds = [record.get("event_type") for record in records]
    for kind in _QWEN_RESUME_CHAIN:
        if kind not in kinds:
            reasons.append(f"missing_{kind}")
    if all(kind in kinds for kind in _QWEN_RESUME_CHAIN) and not (
        has_resume_path(records)
    ):
        reasons.append("broken_resume_chain")
    reasons.extend(resume_stage_reasons(records))
    for record in records:
        for field in _RESOURCE_FIELDS:
            value = record.get(field)
            if value is not None and (type(value) is not int or value < 0):
                reasons.append("resource_value")
        cpu_start = record.get("cpu_process_start_ns")
        cpu_end = record.get("cpu_process_end_ns")
        if cpu_start is not None or cpu_end is not None:
            if (
                type(cpu_start) is not int
                or type(cpu_end) is not int
                or cpu_start < 0
                or cpu_end < cpu_start
            ):
                reasons.append("cpu_process_span")
        if record.get("event_type") in ("kv_export", "kv_import"):
            raw = record.get("raw_bytes")
            transferred = record.get("transferred_bytes")
            if raw is None or transferred is None:
                unknowns.append(
                    f"{record.get('operation_id')}:gpu_transfer_bytes"
                )
            elif raw != transferred and (
                str(record.get("codec")).upper() not in GPU_CODECS
            ):
                # Only a codec that runs on the GPU moves other bytes.
                reasons.append("gpu_transfer_must_equal_raw_bytes")
    return ValidationReport(
        not reasons,
        tuple(dict.fromkeys(reasons)),
        tuple(dict.fromkeys(unknowns)),
    )
