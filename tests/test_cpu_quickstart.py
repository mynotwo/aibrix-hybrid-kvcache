# Copyright 2024 The Aibrix Team.
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

"""Public CPU command acceptance, including no-GPU and no-overwrite behavior."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools.cpu_quickstart import run


def test_cpu_entry_emits_actual_records_without_cuda(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU quickstart entered CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "stream", forbidden)
    output = tmp_path / "example"
    summary = asyncio.run(run(output))
    assert "not hybrid model inference" in summary["scope"]
    assert all(summary["capacity_checks"].values())
    for codec, arm in summary["arms"].items():
        assert arm["raw_bytes"] == 12288
        assert arm["records"] == 2
        assert arm["bytewise_equal"] and arm["trace_valid"]
        assert arm["identity_rejected"] and arm["corruption_rejected"]
        assert arm["dropped_events"] == 0
        assert arm["stored_bytes"] == sum(
            arm[k] for k in ("payload_bytes", "metadata_bytes", "framing_bytes")
        )
        records = [json.loads(line) for line in
                   (output / codec / "events.jsonl").read_text().splitlines()]
        assert len(records) == 8
        assert {r["event_type"] for r in records} == {
            "codec_encode", "l2_put", "l2_get", "codec_decode"
        }
        assert {r["cache_group"] for r in records} == {"state", "span"}
    assert summary["arms"]["identity"]["stored_bytes"] > 12288
    # This fixture is deliberately compressible; it is not a KV-data benchmark.
    assert summary["arms"]["zstd"]["stored_bytes"] < 12288


def test_cli_refuses_to_overwrite_existing_user_artifacts(tmp_path):
    sentinel = tmp_path / "summary.json"
    sentinel.write_text("user result")
    package = Path(__file__).parents[1]
    env = dict(os.environ, PYTHONPATH=str(package))
    result = subprocess.run(
        [sys.executable, "-m", "tools.cpu_quickstart", "--output", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert sentinel.read_text() == "user result"
