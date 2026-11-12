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

from copy import deepcopy

import pytest

from aibrix_kvcache.compression_profiler.recorder import TraceRunStatus
from aibrix_kvcache.compression_profiler.validation import (
    validate_qwen_resume_trace,
)


def _trace():
    kinds = (
        "kv_export",
        "codec_encode",
        "l2_put",
        "eviction",
        "l2_get",
        "codec_decode",
        "kv_import",
    )
    records = []
    for index, kind in enumerate(kinds):
        record = {
            "schema_version": 1,
            "experiment_id": "exp",
            "run_id": "run",
            "request_id": "req",
            "operation_id": kind,
            "dependency_ids": [] if index == 0 else [kinds[index - 1]],
            "parent_operation_id": "",
            "event_type": kind,
            "block_key": "block",
            "generation": 0,
            "cache_group": "5",
            "cache_spec": "qsa",
            "layer_observed": False,
            "start_ns": index * 10,
            "end_ns": index * 10 + 5,
            "result": "restored" if kind == "kv_import" else "ok",
        }
        records.append(record)
    records[0].update(raw_bytes=4096, transferred_bytes=4096)
    records[1].update(
        raw_bytes=4096,
        payload_bytes=2000,
        metadata_bytes=200,
        framing_bytes=12,
        envelope_bytes=2212,
    )
    records[2].update(
        connector_write_bytes=2212, stored_object_bytes=2212
    )
    records[4].update(
        connector_read_bytes=2212, retrieved_object_bytes=2212
    )
    records[5].update(
        raw_bytes=4096,
        decode_output_bytes=4096,
        payload_bytes=2000,
        metadata_bytes=200,
        framing_bytes=12,
        envelope_bytes=2212,
    )
    records[6].update(raw_bytes=4096, transferred_bytes=4096)
    return records


def test_qwen_trace_rejects_missing_operation():
    records = [r for r in _trace() if r["event_type"] != "l2_get"]
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert not report.valid
    assert "missing_l2_get" in report.reasons


def test_qwen_trace_rejects_identity_conflict_and_negative_resource():
    records = _trace()
    records[5]["cache_group"] = "other"
    records[1]["rss_bytes"] = -1
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert "block_identity_conflict" in report.reasons
    assert "resource_value" in report.reasons


def test_qwen_trace_rejects_compressed_size_as_gpu_transfer_bytes():
    records = _trace()
    records[0]["transferred_bytes"] = records[1]["envelope_bytes"]
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert not report.valid
    assert "gpu_transfer_must_equal_raw_bytes" in report.reasons


@pytest.mark.parametrize(
    "codec, valid", [("ANS", True), ("ans", True), ("zstd", False)]
)
def test_only_a_gpu_codec_may_move_its_encoding(codec, valid):
    records = _trace()
    for record in records:
        record["codec"] = codec
        if record["event_type"] in ("kv_export", "kv_import"):
            record["transferred_bytes"] = record["raw_bytes"] // 2
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert ("gpu_transfer_must_equal_raw_bytes" not in report.reasons) == valid


def test_qwen_trace_rejects_negative_cpu_duration_and_bad_envelope_math():
    records = _trace()
    records[1]["cpu_process_start_ns"] = 10
    records[1]["cpu_process_end_ns"] = 9
    records[1]["payload_bytes"] += 1
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert "cpu_process_span" in report.reasons
    assert "envelope_bytes" in report.reasons


def test_qwen_trace_accepts_complete_chain_with_distinct_byte_domains():
    records = deepcopy(_trace())
    records[1].update(
        cpu_process_start_ns=1,
        cpu_process_end_ns=2,
        rss_bytes=1024,
        pinned_bytes=512,
        hbm_allocated_bytes=4096,
        hbm_reserved_bytes=8192,
    )
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert report.valid, report


def test_qwen_trace_follows_dependencies_across_requests():
    """The cold request saves; a later request evicts and restores it."""
    records = deepcopy(_trace())
    for record in records[3:]:
        record["request_id"] = "resumed"
    records[3]["result"] = "local_miss"
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert report.valid, report


def test_qwen_trace_order_comes_from_dependencies_not_file_order():
    # Background threads export events out of causal order.
    records = deepcopy(_trace())
    records = records[3:] + records[:3]
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert report.valid, report


def test_qwen_trace_rejects_restore_not_linked_to_the_save():
    records = deepcopy(_trace())
    records[4]["dependency_ids"] = []
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert not report.valid
    assert "broken_resume_chain" in report.reasons


def test_qwen_trace_rejects_stage_starting_before_its_input_ends():
    records = deepcopy(_trace())
    records[5]["start_ns"] = records[4]["end_ns"] - 1  # decode before get
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert "stage_overlap" in report.reasons


def test_qwen_trace_rejects_encoder_input_other_than_the_export():
    records = deepcopy(_trace())
    records[1]["raw_bytes"] += 1
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert "export_encode_raw_bytes" in report.reasons


def test_qwen_trace_rejects_restore_missing_a_stored_part():
    # A record of two parts (state, span) restored from only one of them.
    records = deepcopy(_trace())
    span = dict(records[2], operation_id="l2_put_span")
    records.append(span)
    records[3]["dependency_ids"] = ["l2_put", "l2_put_span"]
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert "restore_differs_from_store" in report.reasons


def test_qwen_trace_rejects_device_time_longer_than_its_host_window():
    records = deepcopy(_trace())
    records[0].update(enqueue_ns=0, device_start=0, device_end=6)  # host 5
    report = validate_qwen_resume_trace(records, TraceRunStatus())
    assert "device_exceeds_host_window" in report.reasons
