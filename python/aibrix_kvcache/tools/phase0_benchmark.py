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

"""Reproducible streaming benchmark for Phase 0 KV-cache experiments."""

import argparse
import csv
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def gpu_used_bytes() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        used_mib = sum(int(line.strip()) for line in result.stdout.splitlines())
        return used_mib * 2**20
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    total = 0
    pending = [pid]
    seen: set[int] = set()
    try:
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            status = Path(f"/proc/{current}/status").read_text()
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
            children = Path(f"/proc/{current}/task/{current}/children")
            pending.extend(int(value) for value in children.read_text().split())
        return total
    except (FileNotFoundError, PermissionError, ValueError):
        return None


class PeakSampler:
    def __init__(self, pid: int | None) -> None:
        self.pid = pid
        self.peak_hbm: int | None = None
        self.peak_rss: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.05):
            hbm = gpu_used_bytes()
            rss = rss_bytes(self.pid)
            if hbm is not None:
                self.peak_hbm = max(self.peak_hbm or 0, hbm)
            if rss is not None:
                self.peak_rss = max(self.peak_rss or 0, rss)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()


def run_request(
    endpoint: str,
    payload: dict[str, Any],
    label: str,
    iteration: int,
) -> dict[str, Any]:
    payload = payload.copy()
    request_id_prefix = payload.pop("request_id_prefix", label)
    expected_request_id = payload.pop("request_id", None) or (
        f"{request_id_prefix}-{iteration}"
    )
    payload["request_id"] = expected_request_id
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    ttft = None
    token_ids: list[int] = []
    response_request_id = None
    usage = None
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            response_request_id = response_request_id or event.get("id")
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            ids = choices[0].get("token_ids") or []
            if ids and ttft is None:
                ttft = time.perf_counter() - started
            token_ids.extend(ids)
    e2e = time.perf_counter() - started
    return {
        "label": label,
        "iteration": iteration,
        "request_id": response_request_id,
        "requested_request_id": expected_request_id,
        "token_ids": token_ids,
        "output_tokens": len(token_ids),
        "ttft_seconds": ttft,
        "tpot_seconds": (
            (e2e - ttft) / (len(token_ids) - 1)
            if len(token_ids) > 1 and ttft is not None
            else None
        ),
        "e2e_seconds": e2e,
        "tokens_per_second": len(token_ids) / e2e if token_ids else None,
        "usage": usage,
    }


def parse_cache_metrics(log_text: str) -> dict[str, Any]:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log_text)
    reused = [int(value) for value in re.findall(r"reused (\d+) tokens", clean)]
    hits = [int(value) for value in re.findall(r"Num\. of hit: (\d+)", clean)]
    misses = [
        int(value) for value in re.findall(r"Num\. of miss: (\d+)", clean)
    ]
    return {
        "reused_tokens_events": reused,
        "last_cumulative_hit_tokens": hits[-1] if hits else None,
        "last_cumulative_miss_tokens": misses[-1] if misses else None,
    }


def storage_usage(root: Path | None) -> dict[str, int | None]:
    if root is None or not root.exists():
        return {"l2_files": None, "l2_stored_bytes": None}
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "l2_files": len(files),
        "l2_stored_bytes": sum(path.stat().st_size for path in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:18084/v1/completions"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--shfs-root", type=Path)
    parser.add_argument("--prompt-token-ids", type=Path)
    parser.add_argument("--request-id-prefix")
    parser.add_argument("--request-id")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "prompt": (
            json.loads(args.prompt_token_ids.read_text())
            if args.prompt_token_ids
            else args.prompt_file.read_text()
        ),
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "logprobs": 1,
        "request_id_prefix": args.request_id_prefix or args.label,
    }
    if args.request_id:
        if args.iterations != 1 or args.concurrency != 1:
            parser.error("--request-id requires exactly one serial request")
        payload["request_id"] = args.request_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_offset = args.server_log.stat().st_size if args.server_log else 0
    benchmark_started = time.perf_counter()
    with PeakSampler(args.server_pid) as sampler:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    run_request, args.endpoint, payload, args.label, i
                )
                for i in range(args.iterations)
            ]
            records = [future.result() for future in futures]
    benchmark_wall_seconds = time.perf_counter() - benchmark_started

    token_baseline = records[0]["token_ids"]
    for record in records:
        record["token_ids_match_first"] = record["token_ids"] == token_baseline
        record["peak_hbm_bytes"] = sampler.peak_hbm
        record["peak_rss_bytes"] = sampler.peak_rss

    jsonl = args.output_dir / f"{args.label}.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in records))
    fields = [
        "label",
        "iteration",
        "request_id",
        "output_tokens",
        "ttft_seconds",
        "tpot_seconds",
        "e2e_seconds",
        "tokens_per_second",
        "token_ids_match_first",
        "peak_hbm_bytes",
        "peak_rss_bytes",
    ]
    with (args.output_dir / f"{args.label}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log_text = ""
    if args.server_log:
        with args.server_log.open(errors="replace") as file:
            file.seek(log_offset)
            log_text = file.read()
    summary = {
        "label": args.label,
        "iterations": args.iterations,
        "concurrency": args.concurrency,
        "all_token_ids_equal": all(
            row["token_ids_match_first"] for row in records
        ),
        "token_ids": token_baseline,
        "ttft_seconds_mean": (
            statistics.mean(
                row["ttft_seconds"]
                for row in records
                if row["ttft_seconds"] is not None
            )
            if any(row["ttft_seconds"] is not None for row in records)
            else None
        ),
        "e2e_seconds_mean": statistics.mean(
            row["e2e_seconds"] for row in records
        ),
        "benchmark_wall_seconds": benchmark_wall_seconds,
        "throughput_tokens_per_second": (
            sum(row["output_tokens"] for row in records)
            / benchmark_wall_seconds
        ),
        "peak_hbm_bytes": sampler.peak_hbm,
        "peak_rss_bytes": sampler.peak_rss,
        "request_payload": payload,
        "cache_metrics": parse_cache_metrics(log_text),
        "storage": storage_usage(args.shfs_root),
    }
    (args.output_dir / f"{args.label}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
