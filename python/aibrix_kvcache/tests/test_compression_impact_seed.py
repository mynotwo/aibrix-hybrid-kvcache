"""Tests for frozen SHFS seed manifests and isolated file staging."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[1] / "tools" / "compression_impact_seed.py"
PAGE_CACHE_TOOL = TOOL.with_name("compression_impact_page_cache.py")
SPEC = importlib.util.spec_from_file_location("compression_impact_seed", TOOL)
seed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seed
SPEC.loader.exec_module(seed)
PAGE_SPEC = importlib.util.spec_from_file_location(
    "compression_impact_page_cache", PAGE_CACHE_TOOL
)
page_cache = importlib.util.module_from_spec(PAGE_SPEC)
sys.modules[PAGE_SPEC.name] = page_cache
PAGE_SPEC.loader.exec_module(page_cache)


def fixture(tmp_path, storage="raw"):
    source_root = tmp_path / "source-shfs"
    conn_id = seed.connection_id(
        "source", "partition", "block", "builder", "layout"
    )
    key = "aabbccdd"
    source = source_root / conn_id / seed.key_relative_path(key)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable seed block")
    config = tmp_path / "model-config.json"
    prompt = tmp_path / "prompt.txt"
    config.write_text('{"dtype":"bfloat16"}\n')
    prompt.write_text("fixed prompt")
    manifest = tmp_path / f"{storage}.json"
    seed.freeze_seed(
        source_root=source_root,
        output=manifest,
        storage=storage,
        prefix_tokens=128,
        source_namespace="source",
        source_conn_id=conn_id,
        partition_id="partition",
        block_spec_signature="block",
        key_builder_signature="builder",
        layout_signature="layout",
        model="model-x",
        model_revision="revision-x",
        config_path=config,
        prompt_path=prompt,
        sampling_seed=19,
        service_pid_file=tmp_path / "absent.pid",
    )
    return manifest, source


def test_freeze_and_stage_rewrite_only_namespace_and_copy_real_bytes(tmp_path):
    manifest, source = fixture(tmp_path, "identity")
    target_root = tmp_path / "case-shfs"

    staged = seed.stage_manifest(
        manifest,
        target_root=target_root,
        target_namespace="case-unique",
        expected_spec={
            "partition_id": "partition",
            "block_spec_signature": "block",
            "key_builder_signature": "builder",
            "layout_signature": "layout",
        },
    )

    target = Path(staged["entries"][0]["target_path"])
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino
    assert target.parent.parent.parent.name == seed.connection_id(
        "case-unique", "partition", "block", "builder", "layout"
    )
    assert staged["source_manifest_sha256"] == seed.sha256_file(manifest)
    assert staged["entries"][0]["sha256"] == seed.sha256_file(target)


def test_freeze_rejects_symlink_and_unexpected_object_path(tmp_path):
    root = tmp_path / "source-shfs"
    conn_id = seed.connection_id("source", "p", "b", "k", "l")
    object_path = root / conn_id / seed.key_relative_path("aabb")
    object_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"data")
    config = tmp_path / "config.json"
    prompt = tmp_path / "prompt.txt"
    config.write_text("{}\n")
    prompt.write_text("prompt")
    object_path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        seed.freeze_seed(
            source_root=root,
            output=tmp_path / "seed.json",
            storage="raw",
            prefix_tokens=128,
            source_namespace="source",
            source_conn_id=conn_id,
            partition_id="p",
            block_spec_signature="b",
            key_builder_signature="k",
            layout_signature="l",
            model="m",
            model_revision="r",
            config_path=config,
            prompt_path=prompt,
            sampling_seed=0,
            service_pid_file=tmp_path / "absent.pid",
        )


def test_stage_rejects_tampering_spec_mismatch_and_overwrite(tmp_path):
    manifest, source = fixture(tmp_path)
    target_root = tmp_path / "target"
    spec = {
        "partition_id": "partition",
        "block_spec_signature": "block",
        "key_builder_signature": "builder",
        "layout_signature": "layout",
    }
    with pytest.raises(ValueError, match="config mismatch"):
        seed.stage_manifest(
            manifest,
            target_root=target_root,
            target_namespace="case",
            expected_spec={**spec, "layout_signature": "other"},
        )
    frozen = Path(json.loads(manifest.read_text())["source_root"])
    (
        frozen
        / json.loads(manifest.read_text())["source_conn_id"]
        / seed.key_relative_path("aabbccdd")
    ).chmod(0o600)
    (
        frozen
        / json.loads(manifest.read_text())["source_conn_id"]
        / seed.key_relative_path("aabbccdd")
    ).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        seed.stage_manifest(
            manifest,
            target_root=target_root,
            target_namespace="case",
            expected_spec=spec,
        )


def test_stage_rejects_existing_target_and_path_escape(tmp_path):
    manifest, _ = fixture(tmp_path)
    data = json.loads(manifest.read_text())
    data["entries"][0]["object_path"] = "../../escape"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="path"):
        seed.stage_manifest(
            manifest,
            target_root=tmp_path / "target",
            target_namespace="case",
            expected_spec={
                "partition_id": "partition",
                "block_spec_signature": "block",
                "key_builder_signature": "builder",
                "layout_signature": "layout",
            },
        )


def test_stage_rejects_symlinked_parent_inside_frozen_package(tmp_path):
    manifest, _ = fixture(tmp_path)
    data = json.loads(manifest.read_text())
    frozen = Path(data["source_root"])
    conn = frozen / data["source_conn_id"]
    key_path = seed.key_relative_path(data["entries"][0]["key_hex"])
    escaped = tmp_path / key_path.name
    escaped.write_bytes(b"immutable seed block")
    nested = conn / key_path.parent
    nested.rename(nested.with_name(nested.name + "-original"))
    nested.symlink_to(escaped.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="escape|symlink"):
        seed.stage_manifest(
            manifest,
            target_root=tmp_path / "target",
            target_namespace="case",
            expected_spec={
                "partition_id": "partition",
                "block_spec_signature": "block",
                "key_builder_signature": "builder",
                "layout_signature": "layout",
            },
        )


def test_page_cache_control_reports_unknown_when_kernel_probe_is_unavailable(
    tmp_path, monkeypatch
):
    file = tmp_path / "block"
    file.write_bytes(b"block")
    monkeypatch.setattr(page_cache.sys, "platform", "darwin")

    state = page_cache.prepare_page_cache([file], "warm")

    assert state["status"] == "UNKNOWN"
    assert state["verified"] is False
