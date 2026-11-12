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

"""Freeze SHFS object snapshots and stage isolated, verified seed copies."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STORAGE_CODECS = {"raw": "none", "identity": "identity", "zstd": "zstd"}
SIGNATURE_FIELDS = (
    "partition_id",
    "block_spec_signature",
    "key_builder_signature",
    "layout_signature",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connection_id(
    namespace,
    partition_id,
    block_spec_signature,
    key_builder_signature,
    layout_signature,
):
    suffix = f"{block_spec_signature}{key_builder_signature}{layout_signature}"
    return f"{namespace}_{partition_id}_{suffix}"


def key_relative_path(key_hex):
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise ValueError("invalid cache key hex") from exc
    if not key:
        raise ValueError("empty cache key")
    if len(key_hex) >= 4:
        return Path(key_hex[:2]) / key_hex[2:4] / key_hex
    return Path(key_hex)


def _checked_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"seed object is not a regular non-symlink file: {path}"
        )
    return path


def _entry_for(path, base):
    path = _checked_file(path)
    relative = path.relative_to(base)
    key_hex = path.name
    expected = key_relative_path(key_hex)
    if relative != expected:
        raise ValueError(
            f"seed object path does not match key layout: {relative}"
        )
    stat = path.stat()
    if stat.st_size <= 0:
        raise ValueError(f"empty SHFS seed object: {path}")
    return {
        "key_hex": key_hex,
        "object_path": relative.as_posix(),
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def freeze_seed(
    *,
    source_root,
    output,
    storage,
    prefix_tokens,
    source_namespace,
    source_conn_id,
    partition_id,
    block_spec_signature,
    key_builder_signature,
    layout_signature,
    model,
    model_revision,
    config_path,
    prompt_path,
    sampling_seed,
    service_pid_file,
):
    """Snapshot one stopped-task SHFS namespace into immutable seed files."""
    if storage not in STORAGE_CODECS:
        raise ValueError("storage must be raw, identity or zstd")
    if type(prefix_tokens) is not int or prefix_tokens not in (128, 512, 2048):
        raise ValueError("prefix_tokens must be one of 128, 512 or 2048")
    expected_conn_id = connection_id(
        source_namespace,
        partition_id,
        block_spec_signature,
        key_builder_signature,
        layout_signature,
    )
    if source_conn_id != expected_conn_id:
        raise ValueError("source conn_id does not match explicit SHFS config")
    source_root = Path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source SHFS root must be an existing real directory")
    if Path(service_pid_file).exists():
        raise ValueError(
            "service PID file exists; freeze requires operator stop"
        )
    config_path, prompt_path = Path(config_path), Path(prompt_path)
    config = json.loads(config_path.read_text())
    prompt_hash = sha256_file(prompt_path)
    source_namespace_root = source_root / source_conn_id
    if source_namespace_root.is_symlink() or not source_namespace_root.is_dir():
        raise ValueError(
            "source SHFS namespace directory is missing or a symlink"
        )
    entries = []
    seen = set()
    for path in sorted(source_namespace_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in source SHFS namespace: {path}")
        if path.is_file():
            entry = _entry_for(path, source_namespace_root)
            if entry["key_hex"] in seen:
                raise ValueError("duplicate cache key in source SHFS namespace")
            seen.add(entry["key_hex"])
            entries.append((path, entry))
    if not entries:
        raise ValueError("source SHFS namespace has no cache objects")

    output = Path(output)
    package = output.parent / f"{output.stem}.objects"
    package_conn = package / source_conn_id
    if output.exists() or package.exists():
        raise FileExistsError(f"seed snapshot already exists: {package}")
    package_conn.mkdir(parents=True)
    frozen_entries = []
    try:
        for source, entry in entries:
            destination = package_conn / entry["object_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (
                source.open("rb") as source_file,
                destination.open("xb") as target,
            ):
                shutil.copyfileobj(source_file, target)
                target.flush()
                os.fsync(target.fileno())
            if (
                destination.stat().st_size != entry["size"]
                or sha256_file(destination) != entry["sha256"]
            ):
                raise ValueError("seed snapshot copy failed integrity check")
            destination.chmod(0o444)
            frozen_entries.append(entry)
        manifest = {
            "schema_version": 1,
            "storage": storage,
            "codec": STORAGE_CODECS[storage],
            "prefix_tokens": prefix_tokens,
            "source_namespace": source_namespace,
            "source_conn_id": source_conn_id,
            **{
                name: value
                for name, value in zip(
                    SIGNATURE_FIELDS,
                    (
                        partition_id,
                        block_spec_signature,
                        key_builder_signature,
                        layout_signature,
                    ),
                )
            },
            "source_root": str(package.resolve()),
            "model": model,
            "model_revision": model_revision,
            "model_revision_source": "explicit freeze CLI argument",
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "config": config,
            "prompt_path": str(prompt_path.resolve()),
            "prompt_sha256": prompt_hash,
            "sampling_seed": sampling_seed,
            "source_activity": {
                "pid_file": str(Path(service_pid_file).resolve()),
                "pid_file_absent": True,
                "process_inactivity_verified": False,
                "operator_precondition": "source service is stopped",
            },
            "object_boundary": (
                "SHFS backend object bytes; this does not prove "
                "model generation "
                "or a successful KV-cache lifecycle"
            ),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "entries": frozen_entries,
        }
        _write_json_atomic(output, manifest)
        return manifest
    except Exception:
        for path in package_conn.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        shutil.rmtree(package, ignore_errors=True)
        raise


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(path, *, expected_spec, storage=None, prefix_tokens=None):
    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise ValueError("seed manifest cannot be a symlink")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported seed manifest schema")
    if storage is not None and manifest.get("storage") != storage:
        raise ValueError("seed manifest storage mismatch")
    if (
        prefix_tokens is not None
        and manifest.get("prefix_tokens") != prefix_tokens
    ):
        raise ValueError("seed manifest prefix mismatch")
    if any(
        manifest.get(name) != value for name, value in expected_spec.items()
    ):
        raise ValueError("seed manifest config mismatch")
    expected_source_conn = connection_id(
        manifest["source_namespace"],
        *(manifest[name] for name in SIGNATURE_FIELDS),
    )
    if manifest.get("source_conn_id") != expected_source_conn:
        raise ValueError("seed manifest source conn_id mismatch")
    expected_codec = STORAGE_CODECS.get(manifest.get("storage"))
    if manifest.get("codec") != expected_codec:
        raise ValueError("seed manifest codec mismatch")
    if not isinstance(manifest.get("entries"), list) or not manifest["entries"]:
        raise ValueError("seed manifest has no entries")
    return manifest


def stage_manifest(
    path,
    *,
    target_root,
    target_namespace,
    expected_spec,
    storage=None,
    prefix_tokens=None,
):
    manifest = _load_manifest(
        path,
        expected_spec=expected_spec,
        storage=storage,
        prefix_tokens=prefix_tokens,
    )
    target_root = Path(target_root)
    if target_root.exists() and target_root.is_symlink():
        raise ValueError("target SHFS root cannot be a symlink")
    target_root.mkdir(parents=True, exist_ok=True)
    target_conn_id = connection_id(
        target_namespace, *(manifest[name] for name in SIGNATURE_FIELDS)
    )
    conn_root = target_root / target_conn_id
    if conn_root.exists():
        raise FileExistsError(
            f"target SHFS namespace already exists: {conn_root}"
        )
    conn_root.mkdir(parents=True)
    source_root = Path(manifest["source_root"])
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("frozen seed package is missing or a symlink")
    source_conn_root = source_root / manifest["source_conn_id"]
    staged = []
    seen = set()
    try:
        for entry in manifest["entries"]:
            key_hex = entry.get("key_hex")
            relative = key_relative_path(key_hex)
            if entry.get("object_path") != relative.as_posix():
                raise ValueError("seed manifest path does not match cache key")
            if key_hex in seen:
                raise ValueError("duplicate key in seed manifest")
            seen.add(key_hex)
            source = source_conn_root / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError("frozen seed object is missing or a symlink")
            current = source_conn_root
            if current.is_symlink():
                raise ValueError("source namespace contains a symlink")
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("source object path contains a symlink")
            try:
                source.resolve(strict=True).relative_to(
                    source_conn_root.resolve(strict=True)
                )
            except ValueError:
                raise ValueError("source object path escapes seed package")
            if source.stat().st_size != entry.get("size") or sha256_file(
                source
            ) != entry.get("sha256"):
                raise ValueError("seed object hash mismatch")
            target = conn_root / relative
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    f"target seed object already exists: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            created = False
            try:
                with (
                    source.open("rb") as source_file,
                    target.open("xb") as target_file,
                ):
                    created = True
                    shutil.copyfileobj(source_file, target_file)
                    target_file.flush()
                    os.fsync(target_file.fileno())
                if (
                    target.stat().st_size != entry["size"]
                    or sha256_file(target) != entry["sha256"]
                ):
                    raise ValueError("staged seed object hash mismatch")
            except Exception:
                if created:
                    target.unlink(missing_ok=True)
                raise
            if target.stat().st_ino == source.stat().st_ino:
                raise ValueError("seed staging unexpectedly shares an inode")
            staged.append(
                {
                    **entry,
                    "target_path": str(target.resolve()),
                    "source_inode": source.stat().st_ino,
                    "target_inode": target.stat().st_ino,
                }
            )
        result = {
            "schema_version": 1,
            "source_manifest": str(Path(path).resolve()),
            "source_manifest_sha256": sha256_file(path),
            "source_namespace": manifest["source_namespace"],
            "target_namespace": target_namespace,
            "source_conn_id": manifest["source_conn_id"],
            "target_conn_id": target_conn_id,
            "storage": manifest["storage"],
            "prefix_tokens": manifest["prefix_tokens"],
            "entries": staged,
        }
        _write_json_atomic(target_root / "seed-staging.json", result)
        return result
    except Exception:
        shutil.rmtree(conn_root, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    for name in (
        "source-root",
        "output",
        "source-namespace",
        "source-conn-id",
        "partition-id",
        "block-spec-signature",
        "key-builder-signature",
        "layout-signature",
        "model",
        "model-revision",
        "config-file",
        "prompt-file",
        "service-pid-file",
    ):
        freeze.add_argument(f"--{name}", required=True)
    freeze.add_argument("--storage", choices=STORAGE_CODECS, required=True)
    freeze.add_argument(
        "--prefix-tokens", type=int, choices=(128, 512, 2048), required=True
    )
    freeze.add_argument("--sampling-seed", type=int, required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--target-root", type=Path, required=True)
    stage.add_argument("--target-namespace", required=True)
    for name in SIGNATURE_FIELDS:
        flag = "--" + name.replace("_", "-")
        stage.add_argument(flag, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        manifest = freeze_seed(
            source_root=args.source_root,
            output=args.output,
            storage=args.storage,
            prefix_tokens=args.prefix_tokens,
            source_namespace=args.source_namespace,
            source_conn_id=args.source_conn_id,
            partition_id=args.partition_id,
            block_spec_signature=args.block_spec_signature,
            key_builder_signature=args.key_builder_signature,
            layout_signature=args.layout_signature,
            model=args.model,
            model_revision=args.model_revision,
            config_path=args.config_file,
            prompt_path=args.prompt_file,
            sampling_seed=args.sampling_seed,
            service_pid_file=args.service_pid_file,
        )
        print(json.dumps(manifest, indent=2))
    else:
        spec = {name: getattr(args, name) for name in SIGNATURE_FIELDS}
        result = stage_manifest(
            args.manifest,
            target_root=args.target_root,
            target_namespace=args.target_namespace,
            expected_spec=spec,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
