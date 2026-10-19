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
"""Record pieces computed where the bytes are: a checksum that the GPU and
the CPU compute identically, and nvCOMP's ANS codec, which runs on the GPU.
"""

from typing import Iterator, Optional, Sequence

import torch

_ROW = 1024  # int32 words summed per row
# Rows summed at once: widening to int64 takes twice their size in scratch.
_SLICE = 1 << 14


def checksum(flat: torch.Tensor) -> torch.Tensor:
    """Two wrapping int64 sums of ``flat``'s 4-byte words: plain, and
    weighted by row position. They detect corrupted, missing or reordered
    rows of words; they are no defence against deliberate changes."""
    n = flat.numel() // (4 * _ROW) * (4 * _ROW)
    words = flat[:n].view(torch.int32).view(-1, _ROW)
    sums = torch.zeros(2, dtype=torch.int64, device=flat.device)
    sums[0] = flat[n:].sum(dtype=torch.int64)
    for at in range(0, words.shape[0], _SLICE):
        rows = words[at : at + _SLICE].sum(1, dtype=torch.int64)
        weights = torch.arange(
            at + 1, at + 1 + rows.numel(), device=flat.device
        )
        sums += torch.stack([rows.sum(), (rows * weights).sum()])
    return sums


class _TorchBuffer:
    """Device memory for nvCOMP from torch's caching allocator, on the
    current stream (the codec's)."""

    def __init__(self, nbytes: int, stream=None):
        self._tensor = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        self.ptr = self._tensor.data_ptr()


class AnsCodec:
    """nvCOMP ANS on the current CUDA stream, one frame of ``frame_bytes``
    at a time, so its scratch memory does not grow with the record (a
    whole 1.6 GB record needed 3.9 GB). ``nvidia-nvcomp`` is imported only
    when this codec is used. Both directions wait for the stream.
    """

    ALIGN = 256  # nvCOMP decodes only 256-byte aligned frames

    def __init__(self, frame_bytes: int = 64 << 20):
        from nvidia import nvcomp

        # Allocate from torch's pool, not beside it: vLLM leaves the GPU
        # little memory outside the pool, and torch can free cached blocks.
        nvcomp.set_device_allocator(_TorchBuffer)
        self._nvcomp = nvcomp
        self._codecs: dict = {}
        self.frame_bytes = frame_bytes

    def _codec(self, device: torch.device):
        stream = torch.cuda.current_stream(device).cuda_stream
        codec = self._codecs.get(stream)
        if codec is None:
            # Explicit: the calling thread's current device may differ.
            # 16-bit symbols suit bf16 KV and fp32 states alike.
            codec = self._codecs[stream] = self._nvcomp.Codec(
                algorithm="ANS",
                data_type="<f2",
                device_id=device.index,
                cuda_stream=stream,
                uncomp_chunk_size=1 << 18,
            )
        return codec

    def _bytes(self, array, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(array, device=device)
        return tensor.view(torch.uint8).view(-1)

    def encode(self, flat: torch.Tensor) -> Iterator[torch.Tensor]:
        """Each frame of ``flat`` (even length) encoded, zero-padded to
        ``ALIGN`` so frames can be laid end to end; each is valid until the
        next is requested."""
        codec = self._codec(flat.device)
        for at in range(0, flat.numel(), self.frame_bytes):
            frame = flat[at : at + self.frame_bytes]
            encoded = self._bytes(
                codec.encode(self._nvcomp.as_array(frame)), flat.device
            )
            padded = torch.zeros(
                -(-encoded.numel() // self.ALIGN) * self.ALIGN,
                dtype=torch.uint8,
                device=flat.device,
            )
            padded[: encoded.numel()] = encoded
            yield padded

    def decode(
        self,
        payload: torch.Tensor,
        frames: list[int],
        raw_bytes: int,
        ready: Sequence[tuple[int, Optional[torch.cuda.Event]]] = (),
    ) -> torch.Tensor:
        """``payload`` (on the GPU, 256-byte aligned): ``encode``'s frames
        of lengths ``frames``, decoded. ``ready``: in offset order, events
        after which ``payload`` holds its bytes from an offset on; each
        frame waits only for those before its end."""
        codec = self._codec(payload.device)
        stream = torch.cuda.current_stream(payload.device)
        flat = torch.empty(raw_bytes, dtype=torch.uint8, device=payload.device)
        pending = list(ready)
        at = out = 0
        for size in frames:
            while pending and pending[0][0] < at + size:
                event = pending.pop(0)[1]
                if event is not None:
                    stream.wait_event(event)
            frame = self._bytes(
                codec.decode(self._nvcomp.as_array(payload[at : at + size])),
                payload.device,
            )
            if out + frame.numel() > raw_bytes:
                raise ValueError("decoded size mismatch")
            flat[out : out + frame.numel()] = frame
            at, out = at + size, out + frame.numel()
        if out != raw_bytes:
            raise ValueError("decoded size mismatch")
        return flat
