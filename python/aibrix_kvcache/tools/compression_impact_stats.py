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

"""Paired OFF/LIGHT summaries for compression-impact benchmark runs."""

import math
import random
import statistics
from typing import Any

METRIC_LIMITS = {
    "throughput_tokens_per_second": 1.0,
    "ttft_seconds": 1.0,
    "tpot_seconds": 1.0,
    "p95_ttft_seconds": 2.0,
    "p95_tpot_seconds": 2.0,
    "p99_ttft_seconds": None,
    "p99_tpot_seconds": None,
}


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _delta(off, light):
    if not _finite(off) or not _finite(light) or off == 0:
        return None
    return (light - off) / abs(off) * 100.0


def _bootstrap_delta(selected, metric):
    if metric == "throughput_tokens_per_second":
        values = [
            _delta(
                pair["OFF"][metric],
                pair["LIGHT"][metric],
            )
            for pair in selected
        ]
        return (
            statistics.mean(values)
            if all(v is not None for v in values)
            else None
        )
    source = metric.removeprefix("p95_").removeprefix("p99_")
    off = [
        row.get(source) for pair in selected for row in pair["OFF"]["requests"]
    ]
    light = [
        row.get(source)
        for pair in selected
        for row in pair["LIGHT"]["requests"]
    ]
    if not off or not light or not all(_finite(value) for value in off + light):
        return None
    if metric.startswith(("p95_", "p99_")):
        quantile = 0.95 if metric.startswith("p95_") else 0.99
        off_value = percentile(off, quantile)
        light_value = percentile(light, quantile)
    else:
        off_value = statistics.median(off)
        light_value = statistics.median(light)
    return _delta(off_value, light_value)


def summarize_pairs(pairs, *, bootstrap_samples=2000, seed=12345):
    if not pairs:
        raise ValueError("at least one OFF/LIGHT pair is required")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    for pair in pairs:
        if any(
            not pair[mode].get("valid", True)
            or any(not row.get("valid", True) for row in pair[mode]["requests"])
            for mode in ("OFF", "LIGHT")
        ):
            raise ValueError("invalid trace cannot enter statistics")
    rng = random.Random(seed)
    report: dict[str, Any] = {
        "paired_repetitions": len(pairs),
        "batch_concurrency": sorted(
            {len(pair["OFF"]["requests"]) for pair in pairs}
        ),
        "request_observations_per_mode": sum(
            len(pair["OFF"]["requests"]) for pair in pairs
        ),
        "metrics": {},
    }
    for metric, limit in METRIC_LIMITS.items():
        point = _bootstrap_delta(pairs, metric)
        if point is None:
            report["metrics"][metric] = {
                "delta_percent": None,
                "ci95": None,
                "status": "UNAVAILABLE",
            }
            continue
        samples = [
            value
            for sample in (
                rng.choices(pairs, k=len(pairs))
                for _ in range(bootstrap_samples)
            )
            if (value := _bootstrap_delta(sample, metric)) is not None
        ]
        if not samples:
            report["metrics"][metric] = {
                "delta_percent": None,
                "ci95": None,
                "status": "UNAVAILABLE",
            }
            continue
        interval = [percentile(samples, 0.025), percentile(samples, 0.975)]
        crosses_zero = interval[0] <= 0 <= interval[1]
        status = (
            "DESCRIPTIVE"
            if limit is None
            else "INCONCLUSIVE"
            if crosses_zero
            else "PASS"
            if max(abs(interval[0]), abs(interval[1])) <= limit
            else "FAIL"
        )
        report["metrics"][metric] = {
            "delta_percent": point,
            "point_estimate": (
                "mean paired batch delta"
                if metric == "throughput_tokens_per_second"
                else "relative change in pooled median"
                if metric in ("ttft_seconds", "tpot_seconds")
                else "relative change in pooled p95"
                if metric.startswith("p95_")
                else "relative change in pooled p99"
            ),
            "ci95": interval,
            "limit_percent": limit,
            "status": status,
        }
    statuses = [
        item["status"]
        for item in report["metrics"].values()
        if item["status"] != "DESCRIPTIVE"
    ]
    report["status"] = (
        "FAIL"
        if "FAIL" in statuses
        else "INCONCLUSIVE"
        if any(s in ("INCONCLUSIVE", "UNAVAILABLE") for s in statuses)
        else "PASS"
    )
    return report


def summarize_storage_pairs(
    raw_runs, compressed_runs, *, bootstrap_samples=2000, seed=12345
):
    """Describe paired raw/compressed performance without a pass threshold."""
    if len(raw_runs) != len(compressed_runs) or not raw_runs:
        raise ValueError("raw/compressed runs must form non-empty pairs")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    pairs = []
    for raw, compressed in zip(raw_runs, compressed_runs):
        raw_requests = raw.get("requests", [])
        compressed_requests = compressed.get("requests", [])
        if len(raw_requests) != len(compressed_requests):
            raise ValueError("paired request counts differ")
        for raw_request, compressed_request in zip(
            raw_requests, compressed_requests
        ):
            if not raw_request.get("token_ids") or raw_request.get(
                "token_ids"
            ) != compressed_request.get("token_ids"):
                raise ValueError("raw/compressed token IDs differ")
        pairs.append(
            {
                "OFF": {
                    "valid": raw.get("valid", True),
                    "throughput_tokens_per_second": raw.get(
                        "throughput_tokens_per_second"
                    ),
                    "requests": raw_requests,
                },
                "LIGHT": {
                    "valid": compressed.get("valid", True),
                    "throughput_tokens_per_second": compressed.get(
                        "throughput_tokens_per_second"
                    ),
                    "requests": compressed_requests,
                },
            }
        )
    measured = summarize_pairs(
        pairs, bootstrap_samples=bootstrap_samples, seed=seed
    )
    for metric in measured["metrics"].values():
        if metric["status"] != "UNAVAILABLE":
            metric["status"] = "DESCRIPTIVE"
            metric["limit_percent"] = None
    measured["status"] = (
        "INCONCLUSIVE"
        if len(pairs) < 30
        or any(
            metric["status"] == "UNAVAILABLE"
            for metric in measured["metrics"].values()
        )
        else "DESCRIPTIVE"
    )
    return measured


def paired_rows(groups, order):
    pairs = []
    for first, second in ((0, 1), (3, 2)):
        if order[first] == order[second]:
            raise ValueError("ABBA arms did not alternate")
        if len(groups[first]) != len(groups[second]):
            raise ValueError("paired arms must have the same repetition count")
        arms: dict[str, list] = {}
        for left, right in zip(groups[first], groups[second]):
            rows = {
                order[first]: left,
                order[second]: right,
            }
            if not rows["OFF"].get("token_ids") or rows["OFF"].get(
                "token_ids"
            ) != rows["LIGHT"].get("token_ids"):
                raise ValueError(
                    "OFF/LIGHT direct token IDs differ or are unavailable"
                )
            for mode in ("OFF", "LIGHT"):
                arms.setdefault(mode, []).append(stat_row(rows[mode]))
        pairs.append(
            {
                mode: {
                    "valid": True,
                    "throughput_tokens_per_second": groups[
                        first if order[first] == mode else second
                    ][0].get("_batch_throughput_tokens_per_second"),
                    "requests": arms[mode],
                }
                for mode in ("OFF", "LIGHT")
            }
        )
    return pairs


def stat_row(row):
    return {
        "valid": True,
        "throughput_tokens_per_second": row.get("tokens_per_second"),
        "ttft_seconds": row.get("ttft_seconds"),
        "tpot_seconds": row.get("tpot_seconds"),
        "p95_ttft_seconds": row.get("p95_ttft_seconds"),
        "p95_tpot_seconds": row.get("p95_tpot_seconds"),
    }
