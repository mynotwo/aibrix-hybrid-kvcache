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

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import pytest

from aibrix_kvcache.compression_profiler import (
    BlockIdentity,
    EventType,
    ProfileMode,
    TraceContext,
    TraceEvent,
)


def make_event(event_type=EventType.CODEC_DECODE):
    return TraceEvent(
        experiment_id="exp",
        run_id="run",
        request_id="req",
        operation_id="decode-1",
        event_type=event_type,
        parent_operation_id="load-1",
        dependency_ids=("load-1",),
        block_key="aa",
        generation=2,
        layer=3,
        layer_observed=True,
        cache_group="full",
        cache_spec="full_attention",
        source_tier="L2",
        destination_tier="L1",
        phase="reload",
        codec="zstd",
        result="ok",
        error_class="",
        restore_result="restored",
        enqueue_ns=100,
        start_ns=120,
        device_start=None,
        device_end=None,
        end_ns=150,
        raw_bytes=4096,
        payload_bytes=32,
        metadata_bytes=64,
        framing_bytes=12,
        envelope_bytes=108,
        connector_read_bytes=108,
        connector_write_bytes=0,
        workspace_bytes=0,
        checksum_algorithm="sha256",
        checksum_before="abc",
        checksum_after="abc",
        token_count=16,
    )


def test_context_child_preserves_request_and_dependencies():
    root = TraceContext("exp", "run", "req", ProfileMode.DETAILED)
    dependencies = ["op-load"]
    child = root.child("op-decode", dependencies=dependencies)
    dependencies.append("unrelated")
    assert child == TraceContext(
        "exp", "run", "req", ProfileMode.DETAILED,
        "op-decode", "", ("op-load",),
    )
    grandchild = child.child("op-import")
    assert grandchild.parent_operation_id == "op-decode"
    assert grandchild.dependency_ids == ()
    assert root.operation_id == ""


def test_explicit_context_preserves_concurrent_request_identity_in_workers():
    roots = [
        TraceContext("exp", "run", request, ProfileMode.LIGHT)
        for request in ("request-a", "request-b")
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                root.child, f"{root.request_id}-decode",
                dependencies=(f"{root.request_id}-load",),
            )
            for root in roots
        ]
        children = [future.result() for future in futures]
    assert [child.request_id for child in children] == [
        "request-a", "request-b",
    ]
    assert children[0].dependency_ids == ("request-a-load",)
    assert children[1].dependency_ids == ("request-b-load",)
    assert all(child.mode is ProfileMode.LIGHT for child in children)


def test_same_key_generations_remain_distinct():
    a = BlockIdentity("aa", 1, "full", 3)
    b = BlockIdentity("aa", 2, "full", 3)
    assert a != b
    assert len({a, b, replace(a, cache_group="linear"),
                replace(a, layer=4), replace(a, cache_spec="other")}) == 5


@pytest.mark.parametrize("value", [
    TraceContext("exp", "run", "req", ProfileMode.OFF),
    BlockIdentity("aa", 1, "full", 3),
])
def test_context_and_block_identity_are_immutable(value):
    field = "request_id" if isinstance(value, TraceContext) else "block_key"
    with pytest.raises(FrozenInstanceError):
        setattr(value, field, "other")
    assert not hasattr(value, "__dict__")


@pytest.mark.parametrize("event_type", list(EventType))
def test_event_record_uses_controlled_enum_and_schema_version(event_type):
    event = make_event(event_type)
    record = event.to_record()
    assert record["schema_version"] == 1
    assert record["event_type"] == event_type.value
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["dependency_ids"] == ["load-1"]
    record["dependency_ids"].append("unrelated")
    assert event.dependency_ids == ("load-1",)
    with pytest.raises(FrozenInstanceError):
        event.raw_bytes = 0
    assert not hasattr(event, "__dict__")


def test_record_keeps_all_spec_fields_and_arbitrary_storage_tiers():
    record = make_event().to_record()
    assert record == {
        "schema_version": 1,
        "experiment_id": "exp", "run_id": "run", "request_id": "req",
        "operation_id": "decode-1", "parent_operation_id": "load-1",
        "dependency_ids": ["load-1"],
        "block_key": "aa", "generation": 2, "layer": 3,
        "layer_observed": True,
        "cache_group": "full", "cache_spec": "full_attention",
        "source_tier": "L2", "destination_tier": "L1",
        "event_type": "codec_decode", "phase": "reload", "codec": "zstd",
        "result": "ok", "error_class": "", "restore_result": "restored",
        "enqueue_ns": 100, "start_ns": 120, "device_start": None,
        "device_end": None, "cpu_process_start_ns": None,
        "cpu_process_end_ns": None, "end_ns": 150,
        "raw_bytes": 4096, "payload_bytes": 32, "metadata_bytes": 64,
        "framing_bytes": 12, "envelope_bytes": 108,
        "connector_read_bytes": 108, "connector_write_bytes": 0,
        "transferred_bytes": None, "stored_object_bytes": None,
        "retrieved_object_bytes": None, "decode_output_bytes": None,
        "workspace_bytes": 0, "rss_bytes": None, "pinned_bytes": None,
        "hbm_allocated_bytes": None, "hbm_reserved_bytes": None,
        "checksum_algorithm": "sha256",
        "checksum_before": "abc", "checksum_after": "abc", "token_count": 16,
    }
    other = replace(make_event(), source_tier="remote", destination_tier="HBM")
    assert other.to_record()["source_tier"] == "remote"
    assert other.to_record()["destination_tier"] == "HBM"


def test_request_event_preserves_unavailable_cache_time_and_byte_values():
    event = TraceEvent("exp", "run", "req", "request-1", EventType.REQUEST)
    record = event.to_record()
    assert record["block_key"] is None
    assert record["generation"] is None
    assert record["layer"] is None
    assert record["raw_bytes"] is None
    assert record["start_ns"] is None
    assert record["device_start"] is None
    assert record["dependency_ids"] == []
    assert json.loads(json.dumps(record)) == record


def test_mode_and_event_enums_reject_unknown_values():
    assert [mode.value for mode in ProfileMode] == ["off", "light", "detailed"]
    assert {event.value for event in EventType} == {
        "request", "prefill", "decode", "kv_export", "codec_encode", "l2_put",
        "l2_get", "codec_decode", "kv_import", "first_token", "output_token",
        "eviction", "fallback", "error_recovery",
    }
    with pytest.raises(ValueError):
        ProfileMode("unknown")
    with pytest.raises(ValueError):
        EventType("unknown")


def test_direct_context_and_event_copy_dependency_metadata():
    dependencies = ["load-1"]
    context = TraceContext(
        "exp", "run", "req", ProfileMode.LIGHT,
        dependency_ids=dependencies,
    )
    event = replace(make_event(), dependency_ids=dependencies)
    dependencies.append("unrelated")
    assert context.dependency_ids == ("load-1",)
    assert event.dependency_ids == ("load-1",)


@pytest.mark.parametrize("value", [b"payload", object(), {"nested": "value"}])
def test_events_reject_non_scalar_metadata(value):
    with pytest.raises(TypeError):
        replace(make_event(), checksum_before=value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_events_reject_non_json_device_timestamps(value):
    with pytest.raises(ValueError):
        replace(make_event(), device_start=value)


def test_events_reject_unknown_event_type_and_non_string_dependencies():
    with pytest.raises(ValueError):
        replace(make_event(), event_type="unknown")
    with pytest.raises(TypeError):
        replace(make_event(), dependency_ids=(object(),))
