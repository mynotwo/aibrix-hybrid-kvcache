"""Service/request lifecycle for reproducible compression-impact matrices."""

import json
import os
import re
import sys
import uuid
from pathlib import Path

import compression_impact_matrix as matrix

SERVING_DEFAULTS = {
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.55,
    "enforce_eager": True,
    "enable_prefix_caching": False,
}


def overall_status(states):
    states = set(states)
    if "INVALID" in states:
        return "INVALID"
    if "FAIL" in states:
        return "FAIL"
    if "INCONCLUSIVE" in states:
        return "INCONCLUSIVE"
    return "PASS" if states == {"PASS"} else "INVALID"


def validate_serving_config(config, *, model_revision):
    expected = {
        "revision": model_revision,
        **SERVING_DEFAULTS,
    }
    if config != expected:
        raise ValueError("serving config does not match actual runner launch")
    return config


def _serving_env(config):
    return {
        "PHASE0_DTYPE": config["dtype"],
        "PHASE0_TENSOR_PARALLEL_SIZE": str(config["tensor_parallel_size"]),
        "PHASE0_MAX_MODEL_LEN": str(config["max_model_len"]),
        "PHASE0_GPU_MEMORY_UTILIZATION": str(
            config["gpu_memory_utilization"]
        ),
        "PHASE0_ENFORCE_EAGER": str(config["enforce_eager"]).lower(),
        "PHASE0_ENABLE_PREFIX_CACHING": str(
            config["enable_prefix_caching"]
        ).lower(),
    }


def validate_model_provenance(
    model_path, provenance, *, model, revision
):
    model_path = Path(model_path)
    if model_path.is_symlink() or not model_path.is_dir():
        raise ValueError("model path must be a local immutable snapshot")
    resolved = model_path.resolve(strict=True)
    if (
        provenance.get("schema_version") != 1
        or provenance.get("model_repository") != model
        or provenance.get("revision") != revision
        or provenance.get("source", {}).get("kind") != "huggingface_snapshot"
        or resolved.name != revision
        or resolved.parent.name != "snapshots"
        or resolved.parent.parent.name != f"models--{model.replace('/', '--')}"
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError("model provenance is not a matching HF snapshot")
    files = provenance.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model provenance file hash list is required")
    recorded = {}
    for entry in files:
        relative = Path(entry.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model provenance path escapes snapshot")
        path = resolved / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or relative.as_posix() in recorded
        ):
            raise ValueError("model provenance file is missing or repeated")
        digest = matrix.sha256_file(path)
        if digest != entry.get("sha256"):
            raise ValueError(f"model provenance hash mismatch: {relative}")
        recorded[relative.as_posix()] = digest
    weights = {
        path.relative_to(resolved).as_posix()
        for suffix in ("*.safetensors", "*.bin")
        for path in resolved.rglob(suffix)
        if path.is_file() and not path.is_symlink()
    }
    if "config.json" not in recorded or not weights.issubset(recorded):
        raise ValueError("model provenance must cover config and all weights")
    return {
        "model_repository": model,
        "revision": revision,
        "source": provenance["source"],
        "files": recorded,
    }


def run_matrix(args):
    # Calibration must be rejected before any service process is started.
    sys.path.insert(0, str(Path(args.tools_dir).parent))
    from tools.compression_profiler_calibrate import (
        calibration_config,
        code_fingerprint,
        config_fingerprint,
        validate_calibration,
    )

    validate_calibration(
        args.calibration,
        code_hash=code_fingerprint(),
        config_hash=config_fingerprint(calibration_config()),
    )
    vllm_version = matrix._verify_vllm_id_contract(args.venv)
    config = json.loads(args.config_file.read_text())
    serving = validate_serving_config(
        config.get("serving"), model_revision=args.revision
    )
    args.serving_env = _serving_env(serving)
    provenance = json.loads(args.model_provenance.read_text())
    model_provenance = validate_model_provenance(
        args.model_path,
        provenance,
        model=args.model,
        revision=args.revision,
    )
    spec = {
        name: getattr(args, name)
        for name in (
            "partition_id",
            "block_spec_signature",
            "key_builder_signature",
            "layout_signature",
        )
    }
    if config.get("shfs") != spec:
        raise ValueError("config SHFS signatures differ from runtime request")
    manifests = {
        (prefix, storage): args.seed_manifest_dir
        / str(prefix)
        / f"{storage}.json"
        for prefix in matrix.PREFIXES
        for storage in matrix.STORAGES
    }
    seed_hashes = {}
    seed_data = {}
    for (prefix, storage), path in manifests.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required seed manifest missing: {path}")
        data = json.loads(path.read_text())
        if (
            data.get("model") != args.model
            or data.get("model_revision") != args.revision
        ):
            raise ValueError(
                "seed model or revision differs from requested run"
            )
        if data.get("config_sha256") != matrix.sha256_file(args.config_file):
            raise ValueError("seed config hash differs from requested config")
        if data.get("prompt_sha256") != matrix.sha256_file(args.prompt_file):
            raise ValueError("seed prompt hash differs from requested prompt")
        if data.get("sampling_seed") != args.seed:
            raise ValueError("seed sampling seed differs from requested seed")
        if (
            data.get("storage") != storage
            or data.get("prefix_tokens") != prefix
        ):
            raise ValueError("seed storage/prefix manifest path mismatch")
        if any(data.get(key) != value for key, value in spec.items()):
            raise ValueError(
                "seed manifest does not match explicit SHFS config"
            )
        if data.get("config") != config:
            raise ValueError(
                "seed config content differs from requested config"
            )
        seed_hashes[(prefix, storage)] = matrix.sha256_file(path)
        seed_data[(prefix, storage)] = data

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    experiment_id = f"compression-impact-{uuid.uuid4().hex}"
    base_manifest = {
        "model": args.model,
        "model_revision": args.revision,
        "prompt_sha256": matrix.sha256_file(args.prompt_file),
        "config_sha256": matrix.sha256_file(args.config_file),
        "config": config,
        "serving_config": serving,
        "model_provenance": model_provenance,
        "model_provenance_sha256": matrix.sha256_file(
            args.model_provenance
        ),
        "seed_manifest_dir": str(args.seed_manifest_dir.resolve()),
        "seed_manifests": {
            f"{prefix}/{storage}": digest
            for (prefix, storage), digest in seed_hashes.items()
        },
        "sampling_seed": args.seed,
        "sampling": {"temperature": 0, "max_tokens": 32},
        "experiment_id": experiment_id,
        "vllm_version": vllm_version,
        "request_id_contract": (
            "vLLM 0.10.2: cmpl-<request_id>, engine "
            "cmpl-<request_id>-0, n=1"
        ),
    }
    if args.repetitions < 30 or args.repetitions % 2:
        raise ValueError(
            "at least 30 even measured paired repetitions are required"
        )
    summaries = []
    for index, case in enumerate(matrix.build_cases(root, args.seed)):
        case_root = case["output_dir"]
        case_root.mkdir(parents=True, exist_ok=False)
        case["case_root"] = case_root
        blocks = args.repetitions // 2
        token_ids = None
        paired_groups, orders, arm_roots, arm_modes, traces = [], [], [], [], []
        seed_path = manifests[(case["prefix_tokens"], case["storage"])]

        setup_root = case_root / "tokenizer-service"
        setup_root.mkdir()
        setup_env = os.environ.copy()
        setup_env.update(
            {
                "MODEL_PATH": args.model_path,
                "MODEL_REVISION": args.revision,
                "VENV_PATH": args.venv,
                **_serving_env(serving),
            }
        )
        service = [args.bash, str(args.service_script)]
        try:
            matrix._call(
                service
                + [
                    "start",
                    case["storage"],
                    "off",
                    str(setup_root),
                    str(args.port),
                ],
                env=setup_env,
            )
            token_ids = matrix.tokenize_prompt(
                args.endpoint,
                args.model,
                args.prompt_file.read_text(),
                case["prefix_tokens"],
            )
        finally:
            matrix._call(
                service
                + [
                    "stop",
                    case["storage"],
                    "off",
                    str(setup_root),
                    str(args.port),
                ],
                env=setup_env,
            )

        for block in range(blocks):
            order = matrix.pair_order(args.seed + index * blocks + block)
            orders.append(order)
            group_rows = []
            for arm, profiler in enumerate(order):
                group_id = f"b{block}-g{arm}"
                group_rows.append(
                    matrix._run_arm(
                        case,
                        profiler=profiler,
                        iterations=case["concurrency"],
                        warmup=args.warmup,
                        args=args,
                        token_ids=token_ids,
                        group_id=group_id,
                        experiment_id=experiment_id,
                        seed_manifest=seed_path,
                        expected_spec=spec,
                    )
                )
                arm_root = case_root / f"arm-{group_id}-{profiler.lower()}"
                arm_roots.append(arm_root)
                arm_modes.append(profiler)
                traces.append(
                    matrix._validate_trace_if_published(
                        {
                            **case,
                            "trace_path": arm_root
                            / "artifacts/phase0/events.jsonl",
                        },
                        args,
                        profiler,
                        expected_experiment_id=experiment_id,
                        expected_run_id=case["arm_evidence"][group_id][
                            "run_id"
                        ],
                        measured_request_ids=case["arm_evidence"][group_id][
                            "measured_request_ids"
                        ],
                        warmup_request_ids=case["arm_evidence"][group_id][
                            "warmup_request_ids"
                        ],
                        seed_keys={
                            entry["key_hex"]
                            for entry in seed_data[
                                (case["prefix_tokens"], case["storage"])
                            ]["entries"]
                        },
                    )
                )
            paired_groups.extend(matrix._paired_rows(group_rows, order))
        selected_id = f"{case['case_id']}-detailed-selected"
        detail = matrix._run_arm(
            case,
            profiler="DETAILED",
            iterations=1,
            warmup=0,
            args=args,
            token_ids=token_ids,
            group_id="detailed",
            experiment_id=experiment_id,
            seed_manifest=seed_path,
            expected_spec=spec,
            selected_request_id=selected_id,
        )
        arm_roots.append(case_root / "arm-detailed-detailed")
        detail_contract = (
            len(detail) == 1
            and detail[0].get("requested_request_id") == selected_id
            and detail[0].get("request_id") == f"cmpl-{selected_id}"
        )
        detailed_trace = matrix._validate_trace_if_published(
            {
                **case,
                "trace_path": arm_roots[4] / "artifacts/phase0/events.jsonl",
            },
            args,
            "DETAILED",
            expected_experiment_id=experiment_id,
            expected_run_id=case["arm_evidence"]["detailed"]["run_id"],
            measured_request_ids=case["arm_evidence"]["detailed"][
                "measured_request_ids"
            ],
            warmup_request_ids=[],
            seed_keys={
                entry["key_hex"]
                for entry in seed_data[
                    (case["prefix_tokens"], case["storage"])
                ]["entries"]
            },
        )
        if detailed_trace["valid"]:
            events = [
                json.loads(line)
                for line in (arm_roots[4] / "artifacts/phase0/events.jsonl")
                .read_text()
                .splitlines()
            ]
            expected = matrix.detailed_engine_request_id(selected_id)
            if not events or any(
                event.get("request_id") != expected for event in events
            ):
                detailed_trace = {
                    "valid": False,
                    "status": "INVALID",
                    "reason": "DETAILED request identity mismatch",
                }
        traces.extend(
            (
                {
                    "valid": detail_contract,
                    "status": "PASS" if detail_contract else "INVALID",
                    "reason": None
                    if detail_contract
                    else "vLLM response request ID mismatch",
                },
                detailed_trace,
            )
        )
        trace_valid = all(item["valid"] for item in traces)
        stats = (
            matrix.summarize_pairs(
                paired_groups,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + index,
            )
            if trace_valid
            else None
        )
        evidence = list(case.get("arm_evidence", {}).values())
        seed_verified = len(evidence) == 4 * blocks + 1 and all(
            item["seed_staging"] == "VERIFIED" for item in evidence
        )
        cache_verified = len(evidence) == 4 * blocks + 1 and all(
            item.get("page_cache", {}).get("verified") for item in evidence
        )
        unchanged = (
            matrix.sha256_file(seed_path)
            == seed_hashes[(case["prefix_tokens"], case["storage"])]
        )
        case_summary = {
            "case_id": case["case_id"],
            "orders": orders,
            "prefix_tokens": case["prefix_tokens"],
            "concurrency": case["concurrency"],
            "cache_state": case["cache_state"],
            "storage": case["storage"],
            "prompt_token_ids": token_ids,
            "seed_manifest_sha256": seed_hashes[
                (case["prefix_tokens"], case["storage"])
            ],
            "seed_manifest_immutable": unchanged,
            "seed_staging": evidence,
            "page_cache_state": [item.get("page_cache") for item in evidence],
            "trace_validation": traces,
            "statistics": stats,
            "status": stats["status"]
            if stats and seed_verified and cache_verified and unchanged
            else "INVALID",
        }
        (case_root / "summary.json").write_text(
            json.dumps(case_summary, indent=2) + "\n"
        )
        summaries.append(case_summary)
    states = {item["status"] for item in summaries}
    overall = (
        "FAIL"
        if "FAIL" in states
        else "PASS"
        if summaries and states == {"PASS"}
        else "INVALID"
    )
    result = {"manifest": base_manifest, "cases": summaries, "status": overall}
    (root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
