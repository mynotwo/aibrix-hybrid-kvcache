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

"""Exercise real CPU byte storage and codecs; never starts model inference."""

import argparse
import asyncio
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import torch
from aibrix_kvcache.compression import (
    CompressionIdentityError,
    CompressionIntegrityError,
    IdentityCompressionAdapter,
    ZstdCompressionAdapter,
)
from aibrix_kvcache.compression_profiler import EventType, TraceEvent
from aibrix_kvcache.compression_profiler.recorder import ShardedRingRecorder
from aibrix_kvcache.compression_profiler.validation import validate_trace
from aibrix_kvcache.l2.connectors.memory import MemoryConnector


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


async def capacity_checks() -> dict[str, bool]:
    cache = MemoryConnector(capacity_bytes=12)
    cache.open().raise_if_not_ok()
    try:
        for key in (b"a", b"b", b"c"):
            (await cache.put_bytes(key, key * 4)).raise_if_not_ok()
        require(
            (await cache.put_bytes(b"a", b"aaaa")).is_ok(),
            "idempotent write failed",
        )
        require(
            (await cache.put_bytes(b"a", b"other")).is_invalid(),
            "conflicting write was accepted",
        )
        async with cache.lease_bytes(b"b"):
            require(cache.close().is_denied(), "close bypassed active lease")
            require(
                (await cache.put_bytes(b"large", b"x" * 9)).is_out_of_memory(),
                "capacity admission did not reject a protected record",
            )
            for key in (b"a", b"b", b"c"):
                require(
                    (await cache.exists(key)).is_ok(),
                    "rejected admission evicted existing data",
                )
            require(cache.used_bytes == 12, "rejected write changed accounting")
        (await cache.put_bytes(b"fits", b"y" * 8)).raise_if_not_ok()
        require(
            (await cache.get_bytes(b"a")).is_not_found(),
            "oldest eligible record was not evicted",
        )
        require(
            (await cache.get_bytes(b"fits")).get() == b"y" * 8,
            "admitted record was not readable",
        )
    finally:
        cache.close().raise_if_not_ok()
    return dict.fromkeys(
        (
            "idempotent_write",
            "conflicting_write_rejected",
            "lease_protects_data",
            "rejected_admission_preserves_data",
            "post_lease_eviction",
        ),
        True,
    )


async def round_trip(name: str, adapter, output: Path) -> dict:
    # Intentionally repetitive, public synthetic bytes, not model KV tensors.
    samples = {"state": b"state:00" * 512, "span": bytes(range(256)) * 32}
    cache = MemoryConnector(capacity_bytes=32768)
    cache.open().raise_if_not_ok()
    trace_path = output / name / "events.jsonl"
    recorder = ShardedRingRecorder(capacity=32, output=trace_path)
    totals = dict(
        raw_bytes=0,
        payload_bytes=0,
        metadata_bytes=0,
        framing_bytes=0,
        stored_bytes=0,
    )
    guards = dict(identity_rejected=False, corruption_rejected=False)
    try:
        for key, raw in samples.items():

            def event(kind, start, previous="", **fields):
                identifier = f"{key}:{kind.value}"
                require(
                    recorder.record(
                        TraceEvent(
                            "cpu-quickstart",
                            name,
                            "synthetic",
                            identifier,
                            kind,
                            dependency_ids=(previous,) if previous else (),
                            start_ns=start,
                            end_ns=perf_counter_ns(),
                            result="ok",
                            block_key=key,
                            generation=1,
                            cache_group=key,
                            cache_spec="synthetic-byte-vector",
                            codec=name,
                            **fields,
                        )
                    ),
                    "trace recorder rejected an event",
                )
                return identifier

            start = perf_counter_ns()
            block = adapter.encode(
                raw,
                shape=(len(raw),),
                dtype="uint8",
                layout="flat",
                block_key=key,
                generation=1,
                cache_group=key,
            )
            envelope = adapter.serialize(block)
            sizes = adapter.encoded_sizes(block, envelope)
            fields = dict(
                raw_bytes=len(raw),
                payload_bytes=sizes.payload,
                metadata_bytes=sizes.metadata,
                framing_bytes=sizes.framing,
                envelope_bytes=sizes.envelope,
            )
            require(sizes.envelope == len(envelope), "envelope size mismatch")
            encoded = event(EventType.CODEC_ENCODE, start, **fields)
            start = perf_counter_ns()
            (await cache.put_bytes(key.encode(), envelope)).raise_if_not_ok()
            stored = event(
                EventType.L2_PUT,
                start,
                encoded,
                connector_write_bytes=len(envelope),
                **fields,
            )
            start = perf_counter_ns()
            retrieved = bytes((await cache.get_bytes(key.encode())).get())
            loaded = event(
                EventType.L2_GET,
                start,
                stored,
                connector_read_bytes=len(retrieved),
                **fields,
            )
            start = perf_counter_ns()
            restored = adapter.decode(
                adapter.deserialize(retrieved),
                expected_block_key=key,
                expected_generation=1,
                expected_cache_group=key,
                expected_shape=(len(raw),),
                expected_dtype="uint8",
                expected_layout="flat",
            )
            require(restored == raw, "restored bytes differ from input")
            event(
                EventType.CODEC_DECODE,
                start,
                loaded,
                decode_output_bytes=len(restored),
                **fields,
            )
            guards = dict.fromkeys(guards, False)
            try:
                adapter.decode(block, expected_generation=2)
            except CompressionIdentityError:
                guards["identity_rejected"] = True
            corrupt = block._replace(
                metadata=block.metadata._replace(checksum="0" * 64)
            )
            try:
                adapter.decode(corrupt)
            except CompressionIntegrityError:
                guards["corruption_rejected"] = True
            require(all(guards.values()), "identity or checksum guard failed")
            for label, value in (
                ("raw_bytes", len(raw)),
                ("payload_bytes", sizes.payload),
                ("metadata_bytes", sizes.metadata),
                ("framing_bytes", sizes.framing),
                ("stored_bytes", len(envelope)),
            ):
                totals[label] += value
        require(all(guards.values()), "identity or checksum guard failed")
        require(
            cache.used_bytes == totals["stored_bytes"],
            "backend accounting differs from stored envelopes",
        )
    finally:
        recorder.close()
        cache.close().raise_if_not_ok()
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    validation = validate_trace(records, recorder.status, require_complete=True)
    require(validation.valid, f"invalid demo trace: {validation.reasons}")
    require(len(records) == 8, "expected four stages for each of two records")
    return dict(
        **totals,
        **guards,
        records=2,
        trace_events=len(records),
        trace_valid=validation.valid,
        dropped_events=recorder.status.dropped_events,
        raw_to_envelope_ratio=totals["raw_bytes"] / totals["stored_bytes"],
        bytewise_equal=True,
    )


async def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, adapter in (
        ("identity", IdentityCompressionAdapter()),
        ("zstd", ZstdCompressionAdapter()),
    ):
        results[name] = await round_trip(name, adapter, output)
        print(
            f"PASS {name}: 2 records, {results[name]['raw_bytes']} raw bytes, "
            f"{results[name]['stored_bytes']} stored bytes, 8 trace events"
        )
    summary = dict(
        scope="synthetic CPU byte path; not hybrid model inference",
        environment=dict(
            python=sys.version.split()[0],
            torch=torch.__version__,
            platform=platform.platform(),
        ),
        arms=results,
        capacity_checks=await capacity_checks(),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("PASS capacity, duplicate-write, lease, identity and checksum guards")
    print(f"Artifacts: {output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing paths are refused",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
