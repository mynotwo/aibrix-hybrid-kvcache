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
"""Map vLLM KV cache groups to the hybrid record's kinds and block sizes."""

from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from .logic import HybridPlanError

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _layer_kind(spec) -> str:
    # CircularBufferSpec subclasses AttentionSpec, so test it first.
    if isinstance(spec, CircularBufferSpec):
        return "skip"
    if isinstance(spec, MambaSpec):
        if spec.mamba_cache_mode != "align":
            raise HybridPlanError("Mamba layers require mamba_cache_mode=align")
        return "mamba"
    if isinstance(spec, AttentionSpec):
        return "attention"
    raise HybridPlanError(f"unsupported KV cache spec {type(spec).__name__}")


def describe_groups(kv_cache_config: "KVCacheConfig"):
    """Return ({group: {layer: kind}}, {group: kind}, {group: block_size})."""
    layers, kinds, sizes = {}, {}, {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        per_layer = (
            spec.kv_cache_specs
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else {name: spec for name in group.layer_names}
        )
        layers[group_id] = {n: _layer_kind(s) for n, s in per_layer.items()}
        group_kinds = set(layers[group_id].values())
        if len(group_kinds) != 1:
            raise HybridPlanError(f"group {group_id} mixes {group_kinds}")
        kinds[group_id] = group_kinds.pop()
        sizes[group_id] = spec.block_size
    return layers, kinds, sizes
