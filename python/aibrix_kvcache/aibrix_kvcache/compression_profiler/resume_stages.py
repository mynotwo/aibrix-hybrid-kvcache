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
"""Resume-chain rules that byte totals and completion order cannot show."""

from collections import Counter
from collections.abc import Mapping

# Each resume stage starts once the stage it consumes has ended.
_STAGE_INPUT = {
    "codec_encode": "kv_export",
    "l2_put": "codec_encode",
    "l2_get": "eviction",
    "codec_decode": "l2_get",
    "kv_import": "codec_decode",
}


def resume_stage_reasons(records: list[Mapping]) -> list[str]:
    """Stage order, device time within host time, and a restore reading
    back every object it waited for at the size it was stored (a failed
    read counts as a difference)."""
    by_id = {record.get("operation_id"): record for record in records}
    reasons, read = [], {}
    for row in records:
        kind = row.get("event_type")
        # Device work happens inside its host window, queue included.
        if row.get("device_end") is not None:
            begin = row.get("enqueue_ns") or row.get("start_ns", 0)
            if row["device_end"] - row.get("device_start", 0) > row.get(
                "end_ns", 0
            ) - begin:
                reasons.append("device_exceeds_host_window")
        for dep in row.get("dependency_ids") or ():
            prior = by_id.get(dep, {})
            if prior.get("event_type") != _STAGE_INPUT.get(kind):
                continue
            if row.get("start_ns", 0) < prior.get("end_ns", 0):
                reasons.append("stage_overlap")
            if kind == "codec_encode" and row.get("raw_bytes") != prior.get(
                "raw_bytes"
            ):
                reasons.append("export_encode_raw_bytes")
            if kind == "l2_get":
                read.setdefault(dep, Counter())[
                    row.get("retrieved_object_bytes")
                ] += 1
    for eviction, sizes in read.items():
        stored = Counter(
            by_id[dep].get("stored_object_bytes")
            for dep in by_id[eviction].get("dependency_ids") or ()
            if by_id.get(dep, {}).get("event_type") == "l2_put"
        )
        if stored != sizes:
            reasons.append("restore_differs_from_store")
    return reasons
