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

"""Process lifecycle of one matrix case, with every external process faked.

The vLLM service, benchmark client, analyzer and tokenizer endpoint are
replaced, so this pins the commands, environments and evidence the runner
produces, not serving behavior.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import compression_impact_matrix as matrix
from tools import compression_impact_page_cache as page_cache
from tools import compression_impact_runner as runner
from tools import compression_impact_seed as seed
from tools import compression_profiler_calibrate as calibrate

REVISION = "a" * 40
SEED_KEY = "aabbccdd"
SPEC = {
    "partition_id": "partition",
    "block_spec_signature": "block",
    "key_builder_signature": "builder",
    "layout_signature": "layout",
}
CASE = "p128-c1-cold-raw"
TOOLS = Path(matrix.__file__).parent


class FakeHost:
    """Stands in for ``subprocess.run`` and records every command."""

    def __init__(self):
        self.calls = []
        self.profilers = {}

    def run(self, command, **kwargs):
        command = [str(part) for part in command]
        env = kwargs.get("env") or {}
        self.calls.append((command, env))
        done = SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[-1] == "--version":
            done.stdout = "0.10.2"
        elif command[1].endswith("phase0_vllm_service.sh"):
            if command[2] == "start":
                self.profilers[command[5]] = command[4]
        elif command[1].endswith("phase0_benchmark.py"):
            self._benchmark(dict(zip(command[2::2], command[3::2])), env)
        return done

    def _benchmark(self, options, env):
        root, label = Path(options["--output-dir"]), options["--label"]
        profiler = self.profilers[str(root)]
        ids = [options["--request-id"]] if "--request-id" in options else [
            f"{options['--request-id-prefix']}-{i}"
            for i in range(int(options["--iterations"]))
        ]
        rate = 99.5 if profiler == "light" else 100.0
        (root / f"{label}.jsonl").write_text("".join(
            json.dumps({
                "requested_request_id": request,
                "request_id": f"cmpl-{request}",
                "token_ids": [7, 8, 9],
                "ttft_seconds": 10 / rate,
                "tpot_seconds": 1 / rate,
                "tokens_per_second": rate,
            }) + "\n"
            for request in ids
        ))
        (root / f"{label}-summary.json").write_text(
            json.dumps({"throughput_tokens_per_second": rate})
        )
        if profiler == "off":
            return
        trace = root / "artifacts" / "phase0"
        with (trace / "events.jsonl").open("a") as events:
            for request in ids:
                events.write(json.dumps({
                    "experiment_id": env["PHASE0_EXPERIMENT_ID"],
                    "run_id": env["PHASE0_RUN_ID"],
                    "request_id": f"cmpl-{request}-0",
                    "event_type": "l2_get",
                    "result": "hit",
                    "block_key": SEED_KEY,
                }) + "\n")
        (trace / "run-status.json").write_text(
            '{"valid":true,"reasons":[],"dropped_events":0}\n'
        )


def workspace(tmp_path):
    """Model snapshot, config, prompt and one frozen seed per prefix/storage."""
    snapshot = tmp_path / "models--org--model" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "schema_version": 1,
        "model_repository": "org/model",
        "revision": REVISION,
        "source": {"kind": "huggingface_snapshot"},
        "files": [{
            "path": "config.json",
            "sha256": seed.sha256_file(snapshot / "config.json"),
        }],
    }))
    serving = {"revision": REVISION, **runner.SERVING_DEFAULTS}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"serving": serving, "shfs": SPEC}))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixed prompt")
    conn_id = seed.connection_id("source", *SPEC.values())
    source = tmp_path / "source-shfs" / conn_id
    source = source / seed.key_relative_path(SEED_KEY)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"seed block")
    for prefix in (128, 512, 2048):
        for storage in ("raw", "identity", "zstd"):
            seed.freeze_seed(
                source_root=tmp_path / "source-shfs",
                output=tmp_path / "seeds" / str(prefix) / f"{storage}.json",
                storage=storage,
                prefix_tokens=prefix,
                source_namespace="source",
                source_conn_id=conn_id,
                **SPEC,
                model="org/model",
                model_revision=REVISION,
                config_path=config,
                prompt_path=prompt,
                sampling_seed=12345,
                service_pid_file=tmp_path / "absent.pid",
            )
    return snapshot, provenance, config, prompt, serving


def run_first_case(tmp_path, monkeypatch, capsys):
    """Drive the CLI through the first matrix case; return what it did."""
    snapshot, provenance, config, prompt, serving = workspace(tmp_path)
    host = FakeHost()
    class Tokens:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"tokens": list(range(4096))}).encode()

    build_cases = matrix.build_cases
    monkeypatch.setattr(matrix.subprocess, "run", host.run)
    monkeypatch.setattr(
        matrix.urllib.request, "urlopen", lambda *_args, **_kwargs: Tokens()
    )
    monkeypatch.setattr(
        matrix, "build_cases", lambda root, seed: build_cases(root, seed)[:1]
    )
    monkeypatch.setattr(
        calibrate, "validate_calibration", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        page_cache,
        "prepare_page_cache",
        lambda paths, mode, hashes: {
            "mode": mode, "status": "VERIFIED", "verified": True, "files": [],
        },
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [
        "compression_impact_matrix.py",
        "--calibration", str(tmp_path / "calibration.json"),
        "--seed-manifest-dir", str(tmp_path / "seeds"),
        "--model", "org/model",
        "--model-path", str(snapshot),
        "--model-provenance", str(provenance),
        "--revision", REVISION,
        "--partition-id", "partition",
        "--block-spec-signature", "block",
        "--key-builder-signature", "builder",
        "--layout-signature", "layout",
        "--prompt-file", str(prompt),
        "--config-file", str(config),
        "--output-dir", str(tmp_path / "out"),
        "--venv", str(tmp_path / "venv"),
        "--warmup", "1",
        "--bootstrap-samples", "20",
    ])
    matrix.main()
    return SimpleNamespace(
        calls=host.calls,
        printed=json.loads(capsys.readouterr().out),
        root=tmp_path / "out" / CASE,
        snapshot=snapshot,
        provenance=provenance,
        config=config,
        prompt=prompt,
        serving=serving,
    )


def service_command(action, profiler, root):
    script = TOOLS / "phase0_vllm_service.sh"
    return ["bash", str(script), action, "raw", profiler, str(root), "18084"]


def benchmark_command(run, root, label, request_id_prefix, request_id=None):
    command = [
        sys.executable, str(TOOLS / "phase0_benchmark.py"),
        "--endpoint", "http://127.0.0.1:18084/v1/completions",
        "--model", "org/model",
        "--prompt-file", str(run.prompt),
        "--prompt-token-ids", str(root / "prompt-token-ids.json"),
        "--output-dir", str(root),
        "--label", label,
        "--iterations", "1",
        "--concurrency", "1",
        "--max-tokens", "32",
        "--seed", "12345",
        "--request-id-prefix", request_id_prefix,
        "--server-log", str(root / "logs" / "vllm-phase0.log"),
        "--shfs-root", str(root / "shfs"),
    ]
    return command + (["--request-id", request_id] if request_id else [])


def analyzer_command(root):
    trace = root / "artifacts" / "phase0"
    return [
        sys.executable, str(TOOLS / "compression_impact_analyze.py"),
        "--events", str(trace / "events.jsonl"),
        "--run-status", str(trace / "run-status.json"),
        "--output-dir", str(trace / "analysis"),
    ]


SERVING_ENV = {
    "PHASE0_DTYPE": "bfloat16",
    "PHASE0_TENSOR_PARALLEL_SIZE": "1",
    "PHASE0_MAX_MODEL_LEN": "4096",
    "PHASE0_GPU_MEMORY_UTILIZATION": "0.55",
    "PHASE0_ENFORCE_EAGER": "true",
    "PHASE0_ENABLE_PREFIX_CACHING": "false",
}


def test_case_runs_tokenizer_then_abba_arms_then_one_detailed_request(
    tmp_path, monkeypatch, capsys
):
    run = run_first_case(tmp_path, monkeypatch, capsys)
    commands = [command for command, _env in run.calls]
    orders = [matrix.pair_order(12345 + block) for block in range(15)]

    expected = [
        [str(tmp_path / "venv" / "bin" / "vllm"), "--version"],
        service_command("start", "off", run.root / "tokenizer-service"),
        service_command("stop", "off", run.root / "tokenizer-service"),
    ]
    for block, order in enumerate(orders):
        for arm, profiler in enumerate(order):
            group = f"b{block}-g{arm}"
            root = run.root / f"arm-{group}-{profiler.lower()}"
            expected += [
                service_command("start", profiler.lower(), root),
                benchmark_command(
                    run, root, f"{group}-warmup", f"{CASE}-{group}-warmup"
                ),
                benchmark_command(
                    run, root, f"{group}-measured", f"{CASE}-{group}"
                ),
                service_command("stop", profiler.lower(), root),
            ]
            if profiler != "OFF":
                expected.append(analyzer_command(root))
    selected = f"{CASE}-detailed-selected"
    detailed = run.root / "arm-detailed-detailed"
    expected += [
        service_command("start", "detailed", detailed),
        benchmark_command(
            run, detailed, "detailed-measured", selected, selected
        ),
        service_command("stop", "detailed", detailed),
    ]
    # Commands after the DETAILED arm stops validate its trace; the path the
    # runner hands to that validation is not pinned here.
    assert commands[: len(expected)] == expected


def test_service_environments_bind_each_arm_to_its_staged_namespace(
    tmp_path, monkeypatch, capsys
):
    run = run_first_case(tmp_path, monkeypatch, capsys)
    case = run.printed["cases"][0]
    starts = [
        (Path(command[5]), env)
        for command, env in run.calls
        if command[2:3] == ["start"]
    ]

    root, env = starts[0]
    assert root.name == "tokenizer-service"
    assert {name: env[name] for name in SERVING_ENV} == SERVING_ENV
    assert env["MODEL_PATH"] == str(run.snapshot)
    assert env["MODEL_REVISION"] == REVISION
    assert env["VENV_PATH"] == str(tmp_path / "venv")

    assert len(starts) == 1 + len(case["seed_staging"]) == 62
    for (root, env), evidence in zip(starts[1:], case["seed_staging"]):
        assert {name: env[name] for name in SERVING_ENV} == SERVING_ENV
        assert env["MODEL_PATH"] == str(run.snapshot)
        assert env["VENV_PATH"] == str(tmp_path / "venv")
        assert env["PHASE0_NAMESPACE"].startswith(f"phase0-{CASE}-")
        assert env["PHASE0_SHFS_ROOT"] == str(root / "shfs")
        assert env["PHASE0_TRACE_FILE"] == str(
            root / "artifacts" / "phase0" / "l2-trace.jsonl"
        )
        assert env["PHASE0_EXPERIMENT_ID"] == (
            run.printed["manifest"]["experiment_id"]
        )
        assert env["PHASE0_RUN_ID"] == evidence["run_id"]
        assert env["PHASE0_PROMPT_FILE"] == str(run.prompt)
        assert env["PHASE0_EXPECTED_CONN_ID"] == evidence["target_conn_id"]
        assert env["PHASE0_EXPECTED_CONN_ID"] == seed.connection_id(
            env["PHASE0_NAMESPACE"], *SPEC.values()
        )
        assert env["PHASE0_REQUEST_IDS"] == (
            f"cmpl-{CASE}-detailed-selected-0"
            if root.name == "arm-detailed-detailed"
            else ""
        )


def test_case_summary_and_manifest_record_the_evidence_of_every_arm(
    tmp_path, monkeypatch, capsys
):
    run = run_first_case(tmp_path, monkeypatch, capsys)
    out = tmp_path / "out"
    assert run.printed == json.loads((out / "manifest.json").read_text())
    seeds = {
        f"{prefix}/{storage}": seed.sha256_file(
            tmp_path / "seeds" / str(prefix) / f"{storage}.json"
        )
        for prefix in (128, 512, 2048)
        for storage in ("raw", "identity", "zstd")
    }
    manifest = dict(run.printed["manifest"])
    assert manifest.pop("experiment_id").startswith("compression-impact-")
    assert manifest.pop("model_provenance")["files"] == {
        "config.json": seed.sha256_file(run.snapshot / "config.json")
    }
    assert manifest == {
        "model": "org/model",
        "model_revision": REVISION,
        "prompt_sha256": seed.sha256_file(run.prompt),
        "config_sha256": seed.sha256_file(run.config),
        "config": {"serving": run.serving, "shfs": SPEC},
        "serving_config": run.serving,
        "model_provenance_sha256": seed.sha256_file(run.provenance),
        "seed_manifest_dir": str((tmp_path / "seeds").resolve()),
        "seed_manifests": seeds,
        "sampling_seed": 12345,
        "sampling": {"temperature": 0, "max_tokens": 32},
        "vllm_version": "0.10.2",
        "request_id_contract": (
            "vLLM 0.10.2: cmpl-<request_id>, engine cmpl-<request_id>-0, n=1"
        ),
    }

    (case,) = run.printed["cases"]
    assert case == json.loads((run.root / "summary.json").read_text())
    orders = [matrix.pair_order(12345 + block) for block in range(15)]
    groups = [f"b{b}-g{arm}" for b in range(15) for arm in range(4)]
    assert case["case_id"] == CASE
    assert case["orders"] == orders
    assert (case["prefix_tokens"], case["concurrency"]) == (128, 1)
    assert (case["cache_state"], case["storage"]) == ("cold", "raw")
    assert case["prompt_token_ids"] == list(range(128))
    assert case["seed_manifest_sha256"] == seeds["128/raw"]
    assert case["seed_manifest_immutable"] is True
    for group, evidence in zip(groups + ["detailed"], case["seed_staging"]):
        measured = (
            f"{CASE}-detailed-selected"
            if group == "detailed"
            else f"{CASE}-{group}-0"
        )
        assert evidence["seed_staging"] == "VERIFIED"
        assert evidence["seed_staging_sha256"] == seeds["128/raw"]
        assert evidence["page_cache"]["verified"] is True
        assert evidence["run_id"].startswith(f"{CASE}-{group}-")
        assert evidence["measured_request_ids"] == [f"cmpl-{measured}-0"]
        assert evidence["warmup_request_ids"] == (
            []
            if group == "detailed"
            else [f"cmpl-{CASE}-{group}-warmup-0-0"]
        )
    assert case["page_cache_state"] == [
        item["page_cache"] for item in case["seed_staging"]
    ]
    modes = [mode for order in orders for mode in order]
    assert case["trace_validation"][:60] == [
        {"valid": True, "status": "PASS", "reason": "OFF emits no trace"}
        if mode == "OFF"
        else {
            "valid": True,
            "status": "PASS",
            "measured_request_ids_covered": [f"cmpl-{CASE}-{group}-0-0"],
            "warmup_request_ids_observed": [
                f"cmpl-{CASE}-{group}-warmup-0-0"
            ],
            "seed_keys_hit": [SEED_KEY],
        }
        for group, mode in zip(groups, modes)
    ]
    # Entry 61 is the DETAILED trace validation (see the command test).
    assert case["trace_validation"][60] == {
        "valid": True, "status": "PASS", "reason": None,
    }
    assert len(case["trace_validation"]) == 62


def published_trace(tmp_path):
    trace = tmp_path / "events.jsonl"
    trace.write_text("".join(
        json.dumps({
            "experiment_id": "exp",
            "run_id": "run",
            "request_id": request,
            "event_type": "l2_get",
            "result": "hit",
            "block_key": "seed-key",
        }) + "\n"
        for request in ("warmup-engine", "measured-engine")
    ))
    status = tmp_path / "run-status.json"
    status.write_text('{"valid":true,"reasons":[],"dropped_events":0}\n')
    return trace, status


def validate_published(trace, monkeypatch, mode="LIGHT", analyzer=None, **ids):
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda *_args, **_kwargs: analyzer
        or SimpleNamespace(returncode=0, stderr=""),
    )
    return matrix._validate_trace_if_published(
        {"trace_path": trace},
        SimpleNamespace(python="python", analyzer_script=Path("analyze.py")),
        mode,
        **{
            "expected_experiment_id": "exp",
            "expected_run_id": "run",
            "measured_request_ids": ["measured-engine"],
            "warmup_request_ids": ["warmup-engine"],
            "seed_keys": {"seed-key"},
            **ids,
        },
    )


def test_off_arm_passes_only_without_profiler_artifacts(tmp_path, monkeypatch):
    trace = tmp_path / "events.jsonl"
    assert validate_published(trace, monkeypatch, "OFF") == {
        "valid": True, "status": "PASS", "reason": "OFF emits no trace",
    }
    published_trace(tmp_path)
    assert validate_published(trace, monkeypatch, "OFF") == {
        "valid": False,
        "status": "INVALID",
        "reason": "OFF emitted profiler artifacts",
    }


@pytest.mark.parametrize(
    "defect, reason",
    [
        ("no-status", "events or real run-status missing"),
        ("owner-conflict", "multiple recorder owners"),
        ("analyzer", "analysis failed"),
        ("measured-unknown", "measured request identity is missing"),
        ("dropped", "run-status is invalid or dropped events"),
        ("experiment", "trace experiment/run identity mismatch"),
        ("uncovered", "trace does not cover every measured request"),
        ("seed-miss", "measured trace does not prove every staged seed hit"),
    ],
)
def test_published_trace_is_invalid_with_its_reason(
    tmp_path, monkeypatch, defect, reason
):
    trace, status = published_trace(tmp_path)
    analyzer, ids = None, {}
    if defect == "no-status":
        status.unlink()
    elif defect == "owner-conflict":
        (tmp_path / "events.jsonl.owner-conflict").touch()
    elif defect == "analyzer":
        analyzer = SimpleNamespace(returncode=2, stderr=" analysis failed\n")
    elif defect == "measured-unknown":
        ids = {"measured_request_ids": [None]}
    elif defect == "dropped":
        status.write_text('{"valid":true,"reasons":[],"dropped_events":1}\n')
    elif defect == "experiment":
        ids = {"expected_experiment_id": "other"}
    elif defect == "uncovered":
        ids = {"measured_request_ids": ["measured-engine", "absent"]}
    elif defect == "seed-miss":
        ids = {"seed_keys": {"seed-key", "other-key"}}

    result = validate_published(trace, monkeypatch, analyzer=analyzer, **ids)
    assert result == {"valid": False, "status": "INVALID", "reason": reason}
