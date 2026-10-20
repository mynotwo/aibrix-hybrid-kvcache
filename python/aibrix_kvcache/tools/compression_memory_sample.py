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

"""Sample independent process/device/cache counters without summing tiers.

L1/L2 providers report logical resident bytes; RSS/PSS are OS observations;
HBM allocated/reserved are allocator counters, never physical GPU usage or
sample peaks. VmPin is the OS pinned counter, not CUDA allocator accounting.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

COLUMNS = (
    "timestamp_ns",
    "pid",
    "rss_bytes",
    "pss_bytes",
    "pinned_bytes",
    "hbm_allocated_bytes",
    "hbm_reserved_bytes",
    "l1_bytes",
    "l2_bytes",
    "codec_queue_depth",
)


class MemorySampler:
    def __init__(
        self,
        *,
        pid=None,
        mode="light",
        providers=None,
        snapshot=None,
        proc_root="/proc",
    ):
        if mode not in ("light", "detailed"):
            raise ValueError("memory mode must be light or detailed")
        self.pid = os.getpid() if pid is None else pid
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("pid must be positive")
        self.mode, self.providers = mode, dict(providers or {})
        if set(self.providers) - set(COLUMNS[2:]):
            raise ValueError("unknown memory counter")
        self.snapshot = snapshot
        self.proc = Path(proc_root) / str(self.pid)

    def _os_value(self, file, field):
        for line in (self.proc / file).read_text().splitlines():
            key, _, rest = line.partition(":")
            if key == field:
                value, unit = rest.split()
                if unit != "kB":
                    raise ValueError("unknown /proc counter unit")
                return int(value) * 1024
        return None

    def _cuda_value(self, field):
        # PyTorch can only observe its calling process's allocator.
        if self.pid != os.getpid():
            return None
        import torch

        if not torch.cuda.is_available():
            return None
        return getattr(torch.cuda, field)()

    def sample(self, *, phase_boundary=None):
        readers = {
            "rss_bytes": lambda: self._os_value("status", "VmRSS"),
            "pss_bytes": lambda: self._os_value("smaps_rollup", "Pss"),
            "pinned_bytes": lambda: self._os_value("status", "VmPin"),
            "hbm_allocated_bytes": lambda: self._cuda_value("memory_allocated"),
            "hbm_reserved_bytes": lambda: self._cuda_value("memory_reserved"),
        } | self.providers
        values = {"timestamp_ns": time.monotonic_ns(), "pid": self.pid}
        availability, errors = {}, {}
        for name in COLUMNS[2:]:
            try:
                value = readers[name]() if name in readers else None
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("counter must be a nonnegative integer")
            except Exception as exc:
                value = None
                errors[name] = type(exc).__name__
            values[name], availability[name] = value, value is not None
        result = dict(
            values=values,
            availability=availability,
            errors=errors,
            phase_boundary=phase_boundary,
            snapshot=None,
        )
        if self.mode == "detailed" and phase_boundary is not None:
            try:
                if self.pid != os.getpid():
                    raise ValueError("snapshot unavailable for another pid")
                if self.snapshot:
                    result["snapshot"] = self.snapshot()
                else:
                    import torch

                    if torch.cuda.is_available():
                        result["snapshot"] = torch.cuda.memory._snapshot()
            except Exception as exc:
                errors["snapshot"] = type(exc).__name__
        return result

    @staticmethod
    def write_csv(path, samples):
        path, samples = Path(path), list(samples)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    key: "null" if value is None else value
                    for key, value in sample["values"].items()
                }
            )
        data = stream.getvalue().encode()
        metadata = {
            "schema_version": 1,
            "csv_sha256": hashlib.sha256(data).hexdigest(),
            "time_unit": "ns",
            "clock": "CLOCK_MONOTONIC",
            "mode": "point_samples",
            "definitions": {
                "rss_bytes": "/proc/PID/status:VmRSS (kB * 1024)",
                "pss_bytes": "/proc/PID/smaps_rollup:Pss (kB * 1024)",
                "pinned_bytes": "/proc/PID/status:VmPin (kB * 1024)",
                "hbm_allocated_bytes": "current torch CUDA allocator live",
                "hbm_reserved_bytes": "current torch CUDA allocator reserved",
                "l1_bytes": "provider logical resident bytes",
                "l2_bytes": "provider logical resident bytes",
                "codec_queue_depth": "provider queued operations",
            },
            "samples": [
                {key: value for key, value in sample.items() if key != "values"}
                | {
                    "timestamp_ns": sample["values"]["timestamp_ns"],
                    "pid": sample["values"]["pid"],
                }
                for sample in samples
            ],
        }
        sidecar = path.with_suffix(".availability.json")
        encoded = json.dumps(metadata, indent=2, allow_nan=False).encode()
        decode_memory_pair(data, encoded)
        temporary = []
        try:
            for destination, content in ((path, data), (sidecar, encoded)):
                fd, name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=path.parent,
                )
                temporary.append((Path(name), destination))
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            # The sidecar commits the CSV hash. Between replaces the reader
            # rejects mismatched generations, including after a failed replace.
            for source, destination in temporary:
                os.replace(source, destination)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            for source, _ in temporary:
                source.unlink(missing_ok=True)

    @staticmethod
    def read_csv(path):
        """Reject incomplete or mixed generations; never consume CSV alone."""
        path = Path(path)
        sidecar = path.with_suffix(".availability.json")
        try:
            bundle, versions = read_file_bundle((path, sidecar))
            result = decode_memory_pair(bundle[path.name], bundle[sidecar.name])
            verify_file_bundle(versions)
            return result
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid memory publication") from exc


def read_file_bundle(paths):
    """Capture once; callers validate then verify versions before returning."""
    versions = {Path(path): _file_version(Path(path)) for path in paths}
    contents = {path.name: path.read_bytes() for path in versions}
    if len(contents) != len(versions):
        raise ValueError("bundle filenames must be unique")
    return contents, versions


def _file_version(path):
    state = path.stat()
    # Exclude atime; inode/ctime catch replace and same-size writes.
    return tuple(
        getattr(state, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def verify_file_bundle(versions):
    # Check the manifest (the first captured file) last. No sticky invalidation:
    # callers can retry once a writer has committed its whole new generation.
    if any(
        _file_version(path) != versions[path] for path in reversed(versions)
    ):
        raise ValueError("bundle changed; retry committed generation")


def decode_memory_pair(data, encoded):
    """Validate exactly the captured CSV/sidecar bytes, without path rereads."""
    metadata = json.loads(encoded)
    if metadata.get("csv_sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("memory CSV/availability generation mismatch")
    reader = csv.DictReader(io.StringIO(data.decode()))
    if reader.fieldnames != list(COLUMNS):
        raise ValueError("invalid memory CSV columns")
    rows = [
        {
            key: None if value == "null" else int(value)
            for key, value in row.items()
        }
        for row in reader
    ]
    if len(rows) != len(metadata["samples"]):
        raise ValueError("memory sample count mismatch")
    for row, sample in zip(rows, metadata["samples"]):
        if any(sample[key] != row[key] for key in ("timestamp_ns", "pid")):
            raise ValueError("memory sample identity mismatch")
        if set(sample["availability"]) != set(COLUMNS[2:]):
            raise ValueError("incomplete availability map")
        for key, value in row.items():
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid memory counter")
            if key in COLUMNS[2:] and (
                sample["availability"][key] is not (value is not None)
            ):
                raise ValueError("memory null/availability mismatch")
    return dict(rows=rows, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument(
        "--mode", choices=("light", "detailed"), default="light"
    )
    parser.add_argument("--phase-boundary")
    args = parser.parse_args()
    sampler = MemorySampler(pid=args.pid, mode=args.mode)
    sampler.write_csv(
        args.output, [sampler.sample(phase_boundary=args.phase_boundary)]
    )


if __name__ == "__main__":
    main()
