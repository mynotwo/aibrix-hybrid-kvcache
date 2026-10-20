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

"""Accept one real Qwen Next hybrid-connector resume run, or say why not.

Inputs are the compression profiler's per-process outputs
(``<profile>/<scheduler|rankN>/{events.jsonl,run-status.json}``) and the
probe output (``s2-bisect.json``: cold, local_hit and external token IDs).
A run passes only if the merged trace is a valid profiler resume chain
(export -> encode -> put -> eviction -> get -> decode -> import, linked by
dependencies), every rank saved the cold request and restored the external
one, the record equals vLLM's own local-hit blocks on every rank, and the
externally resumed tokens equal vLLM's local prefix-hit resume. Status is
PASS or FAIL; this tool never reports COMPLETE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aibrix_kvcache.compression_profiler.recorder import (  # noqa: E402
    TraceRunStatus,
)
from aibrix_kvcache.compression_profiler.validation import (  # noqa: E402
    has_resume_path,
    validate_qwen_resume_trace,
)

SAVE = ("kv_export", "codec_encode", "l2_put")
SAVED = {"kv_export": "copied", "codec_encode": "encoded", "l2_put": "stored"}


def _of(rows, tag: str, kind: str, phase: str | None = None) -> list[dict]:
    return [
        r
        for r in rows
        # The probe's own first round only, not later benchmark requests.
        if r.get("request_id", "").startswith(f"cmpl-bisect-{tag}-0-")
        and r.get("event_type") == kind
        and (phase is None or r.get("phase") == phase)
    ]


def evaluate_run(
    by_role: dict[str, list[dict]],
    status: TraceRunStatus,
    bisect: dict[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    failures: list[str] = []
    merged = [row for rows in by_role.values() for row in rows]
    report = validate_qwen_resume_trace(merged, status)
    if not report.valid:
        failures.append(f"profiler: {', '.join(report.reasons)}")

    evictions = _of(by_role.get("scheduler", []), "external", "eviction")
    if not evictions or any(e.get("result") != "local_miss" for e in evictions):
        failures.append("eviction: no local miss recorded for the external")
    restores, counts = [], {}
    for rank in range(world_size):
        rows = by_role.get(f"rank{rank}", [])
        # A record is several stored parts (state, spans): check them all.
        for kind in SAVE:
            found = _of(rows, "cold", kind)
            if not found or any(f.get("result") != SAVED[kind] for f in found):
                failures.append(f"save: rank {rank} has no {kind}")
        restored = _of(rows, "external", "kv_import", "restore")
        if not restored or any(r.get("result") != "restored" for r in restored):
            failures.append(f"load: rank {rank} did not restore")
        elif not all(
            has_resume_path(merged, [r["operation_id"]]) for r in restored
        ):
            failures.append(f"profiler: rank {rank} restore chain is broken")
        restores += restored
        # Every rank stores and restores the same parts, one event each.
        counts[rank] = tuple(
            len(_of(rows, tag, kind, phase))
            for tag, kind, phase in [("cold", k, None) for k in SAVE]
            + [("external", "kv_import", "restore")]
        )
        verified = _of(rows, "local", "kv_import", "verify")
        if not verified:
            failures.append(f"bytewise: rank {rank} has no verify (VERIFY off)")
        elif any(v.get("restore_result") != "equal" for v in verified):
            failures.append(
                f"bytewise: rank {rank} record differs from local-hit blocks"
            )
    if len(set(counts.values())) > 1:
        failures.append(f"parts: ranks saved/restored different parts {counts}")
    if any(
        r.get("restore_result") not in ("readback_equal", "restored")
        for r in restores
    ):
        failures.append("readback: restored blocks differ from the record")

    local, external = (
        bisect.get("local_hit") or [],
        bisect.get("external") or [],
    )
    if not local or local != external:
        failures.append(
            "token: external resume must equal local prefix-hit resume"
        )

    # Rank 0's parts (all ranks hold the same amount).
    first = _of(by_role.get("rank0", []), "cold", "codec_encode")
    raw = sum(e.get("raw_bytes") or 0 for e in first) or None
    stored = sum(e.get("envelope_bytes") or 0 for e in first) or None
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "raw_bytes_per_rank": raw,
        "stored_bytes_per_rank": stored,
        "compression_ratio": raw / stored if raw and stored else None,
        "eviction_wait_ns_max": max(
            (e["end_ns"] - e["start_ns"] for e in evictions), default=None
        ),
        "cold_equals_local_hit": bisect.get("cold") == local,
    }


def load_profile(profile: Path) -> tuple[dict, TraceRunStatus]:
    """Events per process role, and the merged run status."""
    by_role, statuses = {}, []
    for events in sorted(profile.glob("*/*.jsonl")):
        by_role[events.parent.name] = [
            json.loads(line)
            for line in events.read_text().splitlines()
            if line.strip()
        ]
        state = json.loads((events.parent / "run-status.json").read_text())
        statuses.append(state)
    status = TraceRunStatus(
        bool(statuses) and all(s.get("valid") for s in statuses),
        tuple(r for s in statuses for r in s.get("reasons", [])),
        sum(s.get("dropped_events", 0) for s in statuses),
    )
    return by_role, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    by_role, status = load_profile(args.run_dir / "profile")
    bisect = json.loads((args.run_dir / "s2-bisect.json").read_text())
    result = evaluate_run(by_role, status, bisect, world_size=args.world_size)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
