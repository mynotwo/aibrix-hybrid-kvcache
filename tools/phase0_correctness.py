#!/usr/bin/env python3
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

"""Validate Phase 0 system artifacts without re-tokenizing response text."""

import argparse
import json
from pathlib import Path

REQUIRED_TRACE_FIELDS = {
    "request_id",
    "operation_id",
    "block_key",
    "generation",
    "layer",
    "cache_group",
    "codec",
    "storage_tier",
    "timestamp",
    "raw_bytes",
    "compressed_bytes",
    "latency",
    "checksum",
    "restore_result",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def tree_usage(path: Path) -> tuple[int, int]:
    files = [entry for entry in path.rglob("*") if entry.is_file()]
    return len(files), sum(entry.stat().st_size for entry in files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--corruption-trace", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--zstd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = [
        "native",
        "raw",
        "zstd",
        "zstd-performance",
        "zstd-cross-process-restore",
        "zstd-corruption-recovery",
        "zstd-concurrency-4",
    ]
    summaries = {
        label: json.loads(
            (args.benchmark_dir / f"{label}-summary.json").read_text()
        )
        for label in labels
    }
    token_ids = {
        label: value["token_ids"] for label, value in summaries.items()
    }
    assert all(token_ids.values()), "direct token_ids must be non-empty"
    assert len({tuple(ids) for ids in token_ids.values()}) == 1, (
        "direct token_ids differ across system modes"
    )
    assert all(value["all_token_ids_equal"] for value in summaries.values())

    trace = read_jsonl(args.trace)
    corruption = read_jsonl(args.corruption_trace)
    assert trace and corruption
    assert all(
        REQUIRED_TRACE_FIELDS <= row.keys() for row in trace + corruption
    )
    results = [row["restore_result"] for row in trace]
    corruption_results = [row["restore_result"] for row in corruption]
    assert "not_found" in results, "fresh L2 miss was not traced"
    assert "restored" in results, "successful L2 restore was not traced"
    assert any(result.startswith("rejected:") for result in corruption_results)
    assert "stored" in corruption_results, "recomputed blocks were not stored"

    raw_files, raw_bytes = tree_usage(args.raw_root)
    zstd_files, zstd_bytes = tree_usage(args.zstd_root)
    assert raw_files == zstd_files > 0
    assert 0 < zstd_bytes < raw_bytes
    report = {
        "status": "PASS",
        "direct_token_ids_equal": True,
        "token_ids": next(iter(token_ids.values())),
        "modes": labels,
        "raw_files": raw_files,
        "raw_bytes": raw_bytes,
        "zstd_files": zstd_files,
        "zstd_bytes": zstd_bytes,
        "storage_ratio_raw_over_zstd": raw_bytes / zstd_bytes,
        "trace_records": len(trace),
        "corruption_trace_records": len(corruption),
        "corruption_rejections": sum(
            result.startswith("rejected:") for result in corruption_results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
