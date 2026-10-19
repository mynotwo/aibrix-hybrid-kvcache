#!/usr/bin/env python3
"""Run isolated OFF/LIGHT compression benchmark pairs and report evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).parent))
import compression_impact_stats as stats  # noqa: E402

PREFIXES = (128, 512, 2048)
CONCURRENCIES = (1, 4, 16)
CACHE_STATES = ("cold", "warm")
STORAGES = ("raw", "identity", "zstd")


def summarize_pairs(*args, **kwargs):
    return stats.summarize_pairs(*args, **kwargs)


def _paired_rows(*args, **kwargs):
    return stats.paired_rows(*args, **kwargs)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_cases(root, seed=12345):
    root = Path(root)
    cases = []
    for prefix in PREFIXES:
        for concurrency in CONCURRENCIES:
            for cache_state in CACHE_STATES:
                for storage in STORAGES:
                    case_id = (
                        f"p{prefix}-c{concurrency}-{cache_state}-{storage}"
                    )
                    case_root = root / case_id
                    cases.append(
                        {
                            "case_id": case_id,
                            "prefix_tokens": prefix,
                            "concurrency": concurrency,
                            "cache_state": cache_state,
                            "page_cache_state": "UNKNOWN",
                            "storage": storage,
                            "seed": seed,
                            "namespace": f"phase0-{case_id}-{uuid.uuid4().hex}",
                            "shfs_root": case_root / "shfs",
                            "trace_path": case_root
                            / "artifacts"
                            / "phase0"
                            / "l2-trace.jsonl",
                            "output_dir": case_root,
                        }
                    )
    return cases


def pair_order(seed):
    import random

    order = ["OFF", "LIGHT", "LIGHT", "OFF"]
    reverse = ["LIGHT", "OFF", "OFF", "LIGHT"]
    return order if random.Random(seed).randrange(2) == 0 else reverse


def detailed_request_ids(request_ids, count):
    if count != 1:
        raise ValueError("DETAILED is restricted to one selected request")
    if len(request_ids) != 1 or not request_ids[0]:
        raise ValueError(
            "DETAILED requires the actual selected vLLM request ID"
        )
    return [request_ids[0]]


def detailed_engine_request_id(request_id, *, n=1):
    if n != 1 or not request_id:
        raise ValueError("verified DETAILED request ID mapping requires n=1")
    return f"cmpl-{request_id}-0"


def engine_request_ids(records):
    result = []
    for row in records:
        requested = row.get("requested_request_id")
        if not requested or row.get("request_id") != f"cmpl-{requested}":
            result.append(None)
        else:
            result.append(detailed_engine_request_id(requested))
    return result


def cache_tier(env):
    """Name the single cache tier a launch env measures; reject overlaps."""
    l1_enabled = env.get("AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED", "0") not in (
        "",
        "0",
        "false",
    )
    backend = env.get("AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND", "")
    if l1_enabled and backend:
        raise ValueError(
            "raw L1 and L2 tiers are mutually exclusive in one benchmark arm"
        )
    if l1_enabled:
        return "RAW_L1"
    if backend == "MEMORY":
        return "LOCAL_COMPRESSED"
    if backend == "SHFS":
        return "SHFS_L2"
    raise ValueError(f"unsupported benchmark cache tier: {backend or 'none'}")


def make_run_manifest(
    *,
    model,
    revision,
    prompt_path,
    config_path,
    seed_manifest,
    sampling_seed,
    cache_env=None,
):
    manifest = {
        "model": model,
        "model_revision": revision,
        "prompt_sha256": sha256_file(prompt_path),
        "config_sha256": sha256_file(config_path),
        "seed_manifest_sha256": sha256_file(seed_manifest),
        "sampling_seed": sampling_seed,
        "sampling": {"temperature": 0, "max_tokens": 32},
        "seed_manifest": str(Path(seed_manifest).resolve()),
        "config": json.loads(Path(config_path).read_text()),
        "prompt": str(Path(prompt_path).resolve()),
    }
    if cache_env is not None:
        manifest["cache_tier"] = cache_tier(cache_env)
    return manifest


def tokenize_prompt(endpoint, model, prompt, prefix_tokens):
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("completion endpoint must be an absolute URL")
    url = urlunsplit((parsed.scheme, parsed.netloc, "/tokenize", "", ""))
    request = urllib.request.Request(
        url,
        data=json.dumps({"model": model, "prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        token_ids = json.loads(response.read()).get("tokens")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) < prefix_tokens
        or any(type(token) is not int for token in token_ids)
    ):
        raise ValueError(
            f"tokenize endpoint did not return {prefix_tokens} token IDs"
        )
    return token_ids[:prefix_tokens]


def _call(command, *, env=None):
    subprocess.run(command, check=True, env=env, timeout=900)


def _run_benchmark(
    case,
    *,
    profiler,
    endpoint,
    model,
    prompt_ids_path,
    benchmark_script,
    python,
    iterations,
    concurrency,
    label,
    server_log,
    request_id_prefix,
    env,
    exact_request_id=None,
):
    command = [
        python,
        str(benchmark_script),
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--prompt-file",
        env["PHASE0_PROMPT_FILE"],
        "--prompt-token-ids",
        str(prompt_ids_path),
        "--output-dir",
        str(case["output_dir"]),
        "--label",
        label,
        "--iterations",
        str(iterations),
        "--concurrency",
        str(concurrency),
        "--max-tokens",
        "32",
        "--seed",
        str(case["seed"]),
        "--request-id-prefix",
        request_id_prefix,
        "--server-log",
        str(server_log),
        "--shfs-root",
        str(case["shfs_root"]),
    ]
    if exact_request_id:
        command.extend(["--request-id", exact_request_id])
    _call(command, env=env)
    return [
        json.loads(line)
        for line in (case["output_dir"] / f"{label}.jsonl")
        .read_text()
        .splitlines()
    ]


def _run_arm(
    case,
    *,
    profiler,
    iterations,
    warmup,
    args,
    token_ids,
    group_id,
    experiment_id,
    seed_manifest,
    expected_spec,
    selected_request_id=None,
):
    if profiler == "DETAILED":
        detailed_request_ids([selected_request_id], count=1)
        filter_id = detailed_engine_request_id(selected_request_id)
    else:
        filter_id = None
    arm_root = case["case_root"] / f"arm-{group_id}-{profiler.lower()}"
    arm_root.mkdir(parents=True, exist_ok=False)
    target_namespace = f"{case['namespace']}-{group_id}-{uuid.uuid4().hex}"
    case["output_dir"] = arm_root
    case["shfs_root"] = arm_root / "shfs"
    artifacts = arm_root / "artifacts" / "phase0"
    artifacts.mkdir(parents=True, exist_ok=True)
    case["trace_path"] = artifacts / "events.jsonl"
    case["l2_trace_path"] = artifacts / "l2-trace.jsonl"
    from compression_impact_page_cache import prepare_page_cache
    from compression_impact_seed import stage_manifest

    staged = stage_manifest(
        seed_manifest,
        target_root=case["shfs_root"],
        target_namespace=target_namespace,
        expected_spec=expected_spec,
        storage=case["storage"],
        prefix_tokens=case["prefix_tokens"],
    )
    case.setdefault("arm_evidence", {})[group_id] = {
        "seed_staging": "VERIFIED",
        "seed_staging_sha256": staged["source_manifest_sha256"],
        "target_conn_id": staged["target_conn_id"],
    }
    token_path = arm_root / "prompt-token-ids.json"
    token_path.write_text(json.dumps(token_ids) + "\n")
    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": args.model_path,
            "VENV_PATH": args.venv,
            "PHASE0_NAMESPACE": target_namespace,
            "PHASE0_SHFS_ROOT": str(case["shfs_root"]),
            "PHASE0_TRACE_FILE": str(case["l2_trace_path"]),
            "PHASE0_REQUEST_IDS": (filter_id or ""),
            "PHASE0_EXPERIMENT_ID": experiment_id,
            "PHASE0_RUN_ID": f"{case['case_id']}-{group_id}-{uuid.uuid4().hex}",
            "PHASE0_PROMPT_FILE": str(args.prompt_file),
            "PHASE0_EXPECTED_CONN_ID": staged["target_conn_id"],
            **args.serving_env,
        }
    )
    service = [args.bash, str(args.service_script)]
    try:
        _call(
            service
            + [
                "start",
                case["storage"],
                profiler.lower(),
                str(arm_root),
                str(args.port),
            ],
            env=env,
        )
        warmup_records = []
        if warmup:
            warmup_records = _run_benchmark(
                case,
                profiler=profiler,
                endpoint=args.endpoint,
                model=args.model,
                prompt_ids_path=token_path,
                benchmark_script=args.benchmark_script,
                python=args.python,
                iterations=warmup,
                concurrency=case["concurrency"],
                label=f"{group_id}-warmup",
                server_log=arm_root / "logs" / "vllm-phase0.log",
                request_id_prefix=f"{case['case_id']}-{group_id}-warmup",
                env=env,
            )
        page_cache = prepare_page_cache(
            [Path(item["target_path"]) for item in staged["entries"]],
            case["cache_state"],
            {
                Path(item["target_path"]): item["sha256"]
                for item in staged["entries"]
            },
        )
        case["arm_evidence"][group_id]["page_cache"] = page_cache
        label = f"{group_id}-measured"
        records = _run_benchmark(
            case,
            profiler=profiler,
            endpoint=args.endpoint,
            model=args.model,
            prompt_ids_path=token_path,
            benchmark_script=args.benchmark_script,
            python=args.python,
            iterations=iterations,
            concurrency=case["concurrency"],
            label=label,
            server_log=arm_root / "logs" / "vllm-phase0.log",
            request_id_prefix=(
                selected_request_id or f"{case['case_id']}-{group_id}"
            ),
            exact_request_id=selected_request_id,
            env=env,
        )
        measured = json.loads((arm_root / f"{label}-summary.json").read_text())
        for record in records:
            record["_batch_throughput_tokens_per_second"] = measured.get(
                "throughput_tokens_per_second"
            )
        case["arm_evidence"][group_id].update(
            {
                "run_id": env["PHASE0_RUN_ID"],
                "measured_request_ids": engine_request_ids(records),
                "warmup_request_ids": engine_request_ids(warmup_records),
            }
        )
        return records
    finally:
        _call(
            service
            + [
                "stop",
                case["storage"],
                profiler.lower(),
                str(arm_root),
                str(args.port),
            ],
            env=env,
        )


def _validate_trace_if_published(
    case,
    args,
    mode,
    *,
    expected_experiment_id=None,
    expected_run_id=None,
    measured_request_ids=(),
    warmup_request_ids=(),
    seed_keys=(),
):
    events = case["trace_path"]
    status = events.parent / "run-status.json"
    if mode == "OFF":
        if events.exists() or status.exists():
            return {
                "valid": False,
                "status": "INVALID",
                "reason": "OFF emitted profiler artifacts",
            }
        return {"valid": True, "status": "PASS", "reason": "OFF emits no trace"}
    if not events.exists() or not status.exists():
        return {
            "valid": False,
            "status": "INVALID",
            "reason": "events or real run-status missing",
        }
    conflict = events.with_suffix(events.suffix + ".owner-conflict")
    if conflict.exists():
        return {
            "valid": False,
            "status": "INVALID",
            "reason": "multiple recorder owners",
        }
    command = [
        args.python,
        str(args.analyzer_script),
        "--events",
        str(events),
        "--run-status",
        str(status),
        "--output-dir",
        str(events.parent / "analysis"),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120
    )
    if result.returncode:
        return {
            "valid": False,
            "status": "INVALID",
            "reason": result.stderr.strip(),
        }
    try:
        rows = [
            json.loads(line)
            for line in events.read_text().splitlines()
            if line.strip()
        ]
        run_status = json.loads(status.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"valid": False, "status": "INVALID", "reason": str(exc)}
    measured = set(measured_request_ids)
    warmup = set(warmup_request_ids)
    seeds = set(seed_keys)
    if None in measured or not measured:
        return {
            "valid": False,
            "status": "INVALID",
            "reason": "measured request identity is missing",
        }
    if not run_status.get("valid") or run_status.get("dropped_events"):
        return {
            "valid": False,
            "status": "INVALID",
            "reason": "run-status is invalid or dropped events",
        }
    if any(
        row.get("experiment_id") != expected_experiment_id
        or row.get("run_id") != expected_run_id
        for row in rows
    ):
        return {
            "valid": False,
            "status": "INVALID",
            "reason": "trace experiment/run identity mismatch",
        }
    observed = {row.get("request_id") for row in rows}
    if not measured.issubset(observed):
        return {
            "valid": False,
            "status": "INVALID",
            "reason": "trace does not cover every measured request",
        }
    seed_hits = {
        row.get("block_key")
        for row in rows
        if row.get("event_type") == "l2_get" and row.get("result") == "hit"
    }
    if seeds and not seeds.issubset(seed_hits):
        return {
            "valid": False,
            "status": "INVALID",
            "reason": "measured trace does not prove every staged seed hit",
        }
    return {
        "valid": True,
        "status": "PASS",
        "measured_request_ids_covered": sorted(measured & observed),
        "warmup_request_ids_observed": sorted(warmup & observed),
        "seed_keys_hit": sorted(seeds & seed_hits),
    }


def _verify_vllm_id_contract(venv):
    binary = Path(venv) / "bin" / "vllm"
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=30
    )
    if result.returncode or "0.10.2" not in result.stdout + result.stderr:
        raise RuntimeError(
            "request-ID trace selection is verified only for vLLM 0.10.2"
        )
    return (result.stdout + result.stderr).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--seed-manifest-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-provenance", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--partition-id", required=True)
    parser.add_argument("--block-spec-signature", required=True)
    parser.add_argument("--key-builder-signature", required=True)
    parser.add_argument("--layout-signature", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--service-script",
        type=Path,
        default=Path(__file__).with_name("phase0_vllm_service.sh"),
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path(__file__).with_name("phase0_benchmark.py"),
    )
    parser.add_argument(
        "--analyzer-script",
        type=Path,
        default=Path(__file__).with_name("compression_impact_analyze.py"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--venv", required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:18084/v1/completions"
    )
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.repetitions < 30:
        parser.error("at least 30 measured paired repetitions are required")
    from compression_impact_runner import run_matrix

    report = run_matrix(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
