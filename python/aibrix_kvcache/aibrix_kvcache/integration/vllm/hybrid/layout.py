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
"""Per-layer view of a hybrid vLLM KV cache, independent of vLLM imports.

Every transferable layer tensor is addressed by ``index_select`` on its block
dimension, so heterogeneous groups (attention, Mamba state tuples) share one
gather/scatter path and no byte offsets are inferred.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

KINDS = ("attention", "mamba", "skip")


class HybridLayoutError(ValueError):
    """The runtime KV cache cannot be described safely."""


@dataclass(frozen=True)
class LayerSlot:
    group_id: int
    name: str
    tensor_index: int
    tensor: torch.Tensor
    block_dim: int

    def describe(self) -> dict:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "tensor_index": self.tensor_index,
            "shape": list(self.tensor.shape),
            "dtype": str(self.tensor.dtype),
            "block_dim": self.block_dim,
        }


def _block_view(
    name: str, tensor: torch.Tensor, num_blocks: int
) -> tuple[torch.Tensor, int]:
    """Return a view whose block dimension has exactly ``num_blocks``.

    Mirrors vLLM's ``group_kernel_blocks``: dim 0 may hold ``ratio`` kernel
    blocks per manager block. Otherwise exactly one dim must equal
    ``num_blocks``.
    """
    leading = tensor.shape[0] if tensor.dim() else 0
    if leading > num_blocks and leading % num_blocks == 0:
        return tensor.unflatten(0, (num_blocks, leading // num_blocks)), 0
    dims = [d for d, size in enumerate(tensor.shape) if size == num_blocks]
    if len(dims) != 1:
        raise HybridLayoutError(
            f"{name}: cannot identify a unique block dimension of size "
            f"{num_blocks} in shape {tuple(tensor.shape)}"
        )
    return tensor, dims[0]


class HybridLayout:
    def __init__(self, slots: Sequence[LayerSlot]):
        self.slots = tuple(slots)
        self.group_ids = tuple(sorted({s.group_id for s in self.slots}))

    @classmethod
    def build(
        cls,
        groups: Mapping[int, Mapping[str, str]],
        kv_caches: Mapping[str, object],
        *,
        num_blocks: int,
    ) -> "HybridLayout":
        """``groups`` maps group id -> {layer name: kind}."""
        slots = []
        for group_id in sorted(groups):
            kinds = set(groups[group_id].values())
            unsupported = kinds - set(KINDS)
            if unsupported:
                raise HybridLayoutError(
                    f"group {group_id}: unsupported layer kind {unsupported}"
                )
            if len(kinds) > 1:
                raise HybridLayoutError(
                    f"group {group_id} mixes layer kinds {sorted(kinds)}"
                )
            if kinds == {"skip"}:
                continue
            for name in sorted(groups[group_id]):
                if name not in kv_caches:
                    raise HybridLayoutError(f"{name}: no KV tensor registered")
                value = kv_caches[name]
                if isinstance(value, (list, tuple)):
                    tensors = tuple(value)
                else:
                    tensors = (value,)
                for index, tensor in enumerate(tensors):
                    view, block_dim = _block_view(name, tensor, num_blocks)
                    slots.append(
                        LayerSlot(group_id, name, index, view, block_dim)
                    )
        if not slots:
            raise HybridLayoutError("layout has nothing to transfer")
        return cls(slots)

    def describe(self) -> list[dict]:
        return [slot.describe() for slot in self.slots]

    def signature(self) -> str:
        payload = json.dumps(self.describe(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def _indices(
        self, block_ids: Mapping[int, Sequence[int]]
    ) -> dict[int, list[int]]:
        if set(block_ids) != set(self.group_ids):
            raise HybridLayoutError(
                f"block ids cover groups {sorted(block_ids)}, "
                f"layout needs {list(self.group_ids)}"
            )
        return {g: list(block_ids[g]) for g in self.group_ids}

    def _check_range(self, slot: LayerSlot, ids: Sequence[int]) -> None:
        size = slot.tensor.shape[slot.block_dim]
        if any(i < 0 or i >= size for i in ids):
            raise HybridLayoutError(f"{slot.name}: block id out of range")

    def _index_tensor(self, slot: LayerSlot, ids: list[int]) -> torch.Tensor:
        # Note: a device index tensor is a host-to-device copy that waits for
        # all work already queued on the stream.
        self._check_range(slot, ids)
        return torch.tensor(ids, dtype=torch.long, device=slot.tensor.device)

    def _picked_shape(self, slot: LayerSlot, count: int) -> list[int]:
        shape = list(slot.tensor.shape)
        shape[slot.block_dim] = count
        return shape

    def gather_sizes(self, block_ids: Mapping[int, Sequence[int]]) -> list[int]:
        """Byte length of each segment ``gather`` would return."""
        ids = self._indices(block_ids)
        return [
            slot.tensor.element_size()
            * int(
                torch.Size(
                    self._picked_shape(slot, len(ids[slot.group_id]))
                ).numel()
            )
            for slot in self.slots
        ]

    def gather(self, block_ids: Mapping[int, Sequence[int]]) -> list[bytes]:
        """Return one raw byte segment per slot, in ``self.slots`` order."""
        ids = self._indices(block_ids)
        segments = []
        for slot in self.slots:
            index = self._index_tensor(slot, ids[slot.group_id])
            picked = slot.tensor.index_select(slot.block_dim, index)
            segments.append(
                picked.contiguous().cpu().view(torch.uint8).numpy().tobytes()
            )
        return segments

    def gather_flat(
        self, block_ids: Mapping[int, Sequence[int]]
    ) -> torch.Tensor:
        """All slots' blocks as one contiguous uint8 tensor on the device."""
        ids = self._indices(block_ids)
        return torch.cat(
            [
                slot.tensor.index_select(
                    slot.block_dim, self._index_tensor(slot, ids[slot.group_id])
                )
                .contiguous()
                .view(torch.uint8)
                .reshape(-1)
                for slot in self.slots
            ]
        )

    def _device_index(
        self, block_ids: Mapping[int, Sequence[int]]
    ) -> dict[int, torch.Tensor]:
        """One device index tensor per group, copied from pinned memory so
        the host does not wait for the work queued on the stream."""
        device = self.slots[0].tensor.device
        index = {}
        for group, ids in self._indices(block_ids).items():
            host = torch.tensor(ids, dtype=torch.long)
            if device.type == "cuda":
                host = host.pin_memory()
            index[group] = host.to(device, non_blocking=True)
        return index

    def _pieces(
        self, block_ids: Mapping[int, Sequence[int]], flat: torch.Tensor
    ) -> list[tuple[LayerSlot, torch.Tensor, torch.Tensor]]:
        """``(slot, index, piece)`` per slot with blocks, ``piece`` being the
        slot's part of ``flat`` (``gather_flat``'s layout) shaped like its
        picked blocks. Checks the total size."""
        ids = self._indices(block_ids)
        sizes = self.gather_sizes(block_ids)
        if flat.numel() != sum(sizes):
            raise HybridLayoutError(
                f"flat record size {flat.numel()} != {sum(sizes)}"
            )
        index = self._device_index(block_ids)
        pieces, offset = [], 0
        for slot, size in zip(self.slots, sizes):
            group_ids = ids[slot.group_id]
            self._check_range(slot, group_ids)
            if group_ids:  # A record part may leave a group out (no bytes).
                piece = (
                    flat[offset : offset + size]
                    .view(slot.tensor.dtype)
                    .view(self._picked_shape(slot, len(group_ids)))
                )
                pieces.append((slot, index[slot.group_id], piece))
            offset += size
        return pieces

    def gather_into(
        self, block_ids: Mapping[int, Sequence[int]], out: torch.Tensor
    ) -> None:
        """``gather_flat`` written into ``out`` (e.g. on the device): one
        kernel per slot, straight from the blocks.

        Copies are only enqueued on the current stream; the caller must wait
        (e.g. on a CUDA event) before reading ``out`` on the host.
        """
        for slot, index, piece in self._pieces(block_ids, out):
            torch.index_select(slot.tensor, slot.block_dim, index, out=piece)

    def scatter_flat(
        self, block_ids: Mapping[int, Sequence[int]], flat: torch.Tensor
    ) -> None:
        """Inverse of ``gather_flat`` for a ``flat`` on the blocks' device;
        checks the total size before writing."""
        for slot, index, piece in self._pieces(block_ids, flat):
            slot.tensor.index_copy_(slot.block_dim, index, piece)

    def scatter(
        self,
        block_ids: Mapping[int, Sequence[int]],
        segments: Sequence[bytes],
    ) -> None:
        """Validate every segment, then write all of them back."""
        ids = self._indices(block_ids)
        if len(segments) != len(self.slots):
            raise HybridLayoutError(
                f"expected {len(self.slots)} segments, got {len(segments)}"
            )
        staged = []
        sizes = self.gather_sizes(block_ids)
        for position, (slot, raw) in enumerate(zip(self.slots, segments)):
            index = self._index_tensor(slot, ids[slot.group_id])
            shape = self._picked_shape(slot, len(index))
            expected = sizes[position]
            if len(raw) != expected:
                raise HybridLayoutError(
                    f"segment {position} size {len(raw)} != {expected}"
                )
            value = (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .view(slot.tensor.dtype)
                .reshape(shape)
            )
            staged.append((slot, index, value))
        for slot, index, value in staged:
            slot.tensor.index_copy_(
                slot.block_dim, index, value.to(slot.tensor.device)
            )
