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
"""Compression-profiler stages of a hybrid record's save and load."""

from typing import Optional

import torch


def stage(trace, kind, key, start, after, **values) -> tuple[str, ...]:
    """Record one stage; its id is what the next stage depends on."""
    if trace is None:
        return ()
    with trace.observation():
        return (trace.event(kind, key, start, after, **values),)
    return ()


def clock(trace) -> int:
    return trace.clock() if trace is not None else 0


def device_event(enabled: bool, timing: bool) -> Optional[torch.cuda.Event]:
    """A CUDA event recorded now on the current stream, or None."""
    if not enabled:
        return None
    event = torch.cuda.Event(enable_timing=timing)
    event.record()
    return event


def device_window(began, ended) -> dict:
    """GPU time between two completed timing events as the device span."""
    if began is None:
        return {}
    return {
        "device_start": 0,
        "device_end": round(began.elapsed_time(ended) * 1_000_000),
    }
