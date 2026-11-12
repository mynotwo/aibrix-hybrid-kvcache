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

"""Behavior tests for the reproducible compression impact matrix."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[1] / "tools" / "compression_impact_matrix.py"
sys.path.insert(0, str(TOOL.parent))
import compression_impact_runner as runner  # noqa: E402
import compression_impact_stats as impact_stats  # noqa: E402

SPEC = importlib.util.spec_from_file_location("compression_impact_matrix", TOOL)
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "phase0_benchmark", TOOL.with_name("phase0_benchmark.py")
)
benchmark = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = benchmark
BENCHMARK_SPEC.loader.exec_module(benchmark)


def test_frozen_cases_have_isolated_namespace_and_trace_paths(tmp_path):
    cases = matrix.build_cases(tmp_path, seed=42)

    assert len(cases) == 3 * 3 * 2 * 3
    assert len({case["namespace"] for case in cases}) == len(cases)
    assert len({case["trace_path"] for case in cases}) == len(cases)
    assert all(case["shfs_root"].is_relative_to(tmp_path) for case in cases)
    assert {
        (
            case["prefix_tokens"],
            case["concurrency"],
            case["cache_state"],
            case["storage"],
        )
        for case in cases
    } == {
        (prefix, concurrency, cache, storage)
        for prefix in (128, 512, 2048)
        for concurrency in (1, 4, 16)
        for cache in ("cold", "warm")
        for storage in ("raw", "identity", "zstd")
    }


def test_pair_order_is_reproducible_abba_and_detailed_is_single_request():
    first = matrix.pair_order(seed=314)
    assert first == ["OFF", "LIGHT", "LIGHT", "OFF"]
    assert matrix.pair_order(seed=315) == ["LIGHT", "OFF", "OFF", "LIGHT"]
    assert matrix.detailed_request_ids(["request-from-vllm"], count=1) == [
        "request-from-vllm"
    ]
    assert matrix.detailed_engine_request_id("request-from-vllm") == (
        "cmpl-request-from-vllm-0"
    )
    with pytest.raises(ValueError, match="n=1"):
        matrix.detailed_engine_request_id("request-from-vllm", n=2)


def test_manifest_binds_seed_prompt_revision_and_config(tmp_path):
    seed = tmp_path / "seed.json"
    prompt = tmp_path / "prompt.txt"
    config = tmp_path / "config.json"
    seed.write_text('{"blocks": ["immutable-block"]}\n')
    prompt.write_text("fixed prompt")
    config.write_text('{"dtype":"bfloat16"}\n')

    manifest = matrix.make_run_manifest(
        model="model-a",
        revision="rev-123",
        prompt_path=prompt,
        config_path=config,
        seed_manifest=seed,
        sampling_seed=9,
    )

    assert manifest["model_revision"] == "rev-123"
    assert manifest["sampling_seed"] == 9
    assert manifest["prompt_sha256"] == matrix.sha256_file(prompt)
    assert manifest["seed_manifest_sha256"] == matrix.sha256_file(seed)
    assert manifest["config_sha256"] == matrix.sha256_file(config)


def test_manifest_cache_tier_keeps_raw_l1_and_local_compressed_exclusive():
    assert matrix.cache_tier({"AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED": "1"}) == (
        "RAW_L1"
    )
    assert (
        matrix.cache_tier(
            {
                "AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED": "0",
                "AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND": "MEMORY",
            }
        )
        == "LOCAL_COMPRESSED"
    )
    assert (
        matrix.cache_tier(
            {
                "AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED": "0",
                "AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND": "SHFS",
            }
        )
        == "SHFS_L2"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        matrix.cache_tier(
            {
                "AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED": "1",
                "AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND": "MEMORY",
            }
        )


def test_manifest_records_cache_tier_from_launch_env(tmp_path):
    for name in ("seed.json", "prompt.txt", "config.json"):
        (tmp_path / name).write_text("{}\n")

    manifest = matrix.make_run_manifest(
        model="model-a",
        revision="rev-123",
        prompt_path=tmp_path / "prompt.txt",
        config_path=tmp_path / "config.json",
        seed_manifest=tmp_path / "seed.json",
        sampling_seed=9,
        cache_env={
            "AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED": "0",
            "AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND": "MEMORY",
        },
    )

    assert manifest["cache_tier"] == "LOCAL_COMPRESSED"


def test_tokenize_prompt_uses_vllm_root_tokenize_path(monkeypatch):
    requested = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"tokens": list(range(128))}).encode()

    def fake_urlopen(request, timeout):
        requested.append(request.full_url)
        return Response()

    monkeypatch.setattr(matrix.urllib.request, "urlopen", fake_urlopen)
    assert matrix.tokenize_prompt(
        "http://127.0.0.1:18084/v1/completions", "model", "prompt", 128
    ) == list(range(128))
    assert requested == ["http://127.0.0.1:18084/tokenize"]


def test_trace_validation_requires_measured_run_identity_and_seed_hits(
    tmp_path, monkeypatch
):
    trace = tmp_path / "events.jsonl"
    status = tmp_path / "run-status.json"
    rows = [
        {
            "experiment_id": "exp",
            "run_id": "run",
            "request_id": request_id,
            "event_type": "l2_get",
            "result": "hit",
            "block_key": "seed-key",
        }
        for request_id in ("warmup-engine", "measured-engine")
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    status.write_text('{"valid":true,"reasons":[],"dropped_events":0}\n')
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Done", (), {"returncode": 0, "stderr": ""}
        )(),
    )
    result = matrix._validate_trace_if_published(
        {"trace_path": trace},
        type("Args", (), {
            "python": "python",
            "analyzer_script": tmp_path / "analyze.py",
        })(),
        "LIGHT",
        expected_experiment_id="exp",
        expected_run_id="run",
        measured_request_ids=["measured-engine"],
        warmup_request_ids=["warmup-engine"],
        seed_keys={"seed-key"},
    )
    assert result["valid"] is True
    assert result["measured_request_ids_covered"] == ["measured-engine"]


def test_runtime_config_and_model_provenance_are_verified(tmp_path):
    snapshot = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"test"}\n')
    (snapshot / "weights.safetensors").write_bytes(b"weights")
    provenance = {
        "schema_version": 1,
        "model_repository": "org/model",
        "revision": "a" * 40,
        "source": {"kind": "huggingface_snapshot"},
        "files": [
            {
                "path": name,
                "sha256": matrix.sha256_file(snapshot / name),
            }
            for name in ("config.json", "weights.safetensors")
        ],
    }
    serving = {
        "revision": "a" * 40,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.55,
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }
    assert runner.validate_model_provenance(
        snapshot, provenance, model="org/model", revision="a" * 40
    )["revision"] == "a" * 40
    assert (
        runner.validate_serving_config(serving, model_revision="a" * 40)
        == serving
    )
    with pytest.raises(ValueError, match="serving config"):
        runner.validate_serving_config(
            {**serving, "tensor_parallel_size": 2},
            model_revision="a" * 40,
        )


def test_storage_pair_summary_preserves_token_ids_and_inconclusive_status():
    raw = [
        {
            "throughput_tokens_per_second": 100,
            "requests": [{"token_ids": [1], "ttft_seconds": 1.0}],
        }
        for _ in range(30)
    ]
    zstd = [
        {
            "throughput_tokens_per_second": value,
            "requests": [{"token_ids": [1], "ttft_seconds": 1.0}],
        }
        for value in (98, 102) * 15
    ]
    result = impact_stats.summarize_storage_pairs(
        raw, zstd, bootstrap_samples=200, seed=2
    )
    assert result["paired_repetitions"] == 30
    assert result["metrics"]["throughput_tokens_per_second"]["status"] == (
        "DESCRIPTIVE"
    )
    with pytest.raises(ValueError, match="token IDs differ"):
        impact_stats.summarize_storage_pairs(raw, [{**zstd[0], "requests": [
            {"token_ids": [2], "ttft_seconds": 1.0}
        ]}] * 30)


def test_overall_status_preserves_inconclusive_and_invalid_states():
    assert runner.overall_status(["PASS", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert runner.overall_status(["PASS", "INVALID"]) == "INVALID"
    assert runner.overall_status(["PASS", "FAIL"]) == "FAIL"


def test_statistics_keep_missing_metrics_null_and_crossing_ci_inconclusive():
    result = matrix.summarize_pairs(
        [
            {
                "OFF": {
                    "valid": True,
                    "throughput_tokens_per_second": 100.0,
                    "requests": [
                        {
                            "valid": True,
                            "ttft_seconds": None,
                            "tpot_seconds": None,
                        }
                    ],
                },
                "LIGHT": {
                    "valid": True,
                    "throughput_tokens_per_second": value,
                    "requests": [
                        {
                            "valid": True,
                            "ttft_seconds": None,
                            "tpot_seconds": None,
                        }
                    ],
                },
            }
            for value in (98.0, 102.0) * 15
        ],
        bootstrap_samples=500,
        seed=2,
    )

    assert result["metrics"]["ttft_seconds"]["delta_percent"] is None
    assert result["metrics"]["ttft_seconds"]["ci95"] is None
    assert (
        result["metrics"]["throughput_tokens_per_second"]["status"]
        == "INCONCLUSIVE"
    )


def test_invalid_trace_is_rejected_from_statistics():
    pairs = [
        {
            "OFF": {
                "valid": True,
                "throughput_tokens_per_second": 10.0,
                "requests": [],
            },
            "LIGHT": {
                "valid": False,
                "throughput_tokens_per_second": 11.0,
                "requests": [],
            },
        }
    ]
    with pytest.raises(ValueError, match="invalid trace"):
        matrix.summarize_pairs(pairs, bootstrap_samples=100, seed=1)


def test_streaming_request_uses_and_preserves_direct_token_ids(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(
                [
                    (
                        b'data: {"id":"cmpl-case-0","choices":'
                        b'[{"token_ids":[11,12]}]}\n'
                    ),
                    b"data: [DONE]\n",
                ]
            )

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    record = benchmark.run_request(
        "http://fake/v1/completions", {"request_id_prefix": "case"}, "case", 0
    )

    assert captured["payload"]["request_id"] == "case-0"
    assert record["requested_request_id"] == "case-0"
    assert record["request_id"] == "cmpl-case-0"
    assert record["token_ids"] == [11, 12]


def test_streaming_request_can_use_exact_selected_id_for_detailed_trace(
    monkeypatch,
):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter([b'data: {"id":"cmpl-selected","choices":[]}\n'])

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    record = benchmark.run_request(
        "http://fake/v1/completions",
        {"request_id": "selected"},
        "detailed",
        0,
    )

    assert captured["payload"]["request_id"] == "selected"
    assert "request_id_prefix" not in captured["payload"]
    assert record["requested_request_id"] == "selected"
    assert record["request_id"] == "cmpl-selected"


def test_missing_stream_tokens_remain_null_metrics(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter([b'data: {"id":"cmpl-case-0","choices":[]}\n'])

    monkeypatch.setattr(
        benchmark.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    record = benchmark.run_request("http://fake/v1/completions", {}, "case", 0)

    assert record["ttft_seconds"] is None
    assert record["tpot_seconds"] is None
    assert record["tokens_per_second"] is None
