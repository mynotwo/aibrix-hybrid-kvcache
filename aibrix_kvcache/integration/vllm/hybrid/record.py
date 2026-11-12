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
"""Record bytes: MAGIC | u32 header length | JSON header | payload.

The payload is the layout's flat bytes (IDENTITY), or independent frames,
each nvCOMP's ANS encoding of ``chunk_bytes`` of them, made and decoded on
the GPU (ANS). The header carries ``gpu_codec.checksum`` of the flat
bytes, which a load checks on the GPU after the bytes got there.

A load stages a payload that is not page-locked in a pinned host buffer,
in parallel chunks, each usable as soon as it is staged.
"""

import json
import struct
import warnings
import weakref
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

MAGIC = b"AIBXHR3\n"
FRAMING = len(MAGIC) + 4
CODECS = ("IDENTITY", "ANS")
CHECKSUMS = {"IDENTITY": "word-sums", "ANS": "word-sums"}


@dataclass(frozen=True)
class Header:
    raw_bytes: int
    checksum: list[int]
    meta_bytes: int
    payload: memoryview
    frames: list[int]  # payload lengths of the frames, if framed
    chunk_bytes: int  # raw bytes per frame, if framed


def pin(envelope: memoryview) -> None:
    """Page-lock an envelope from ``RecordCodec.encode`` until it is freed,
    so the GPU reads it by DMA (best effort)."""
    array = cast(np.ndarray, envelope.obj)
    cudart, at = torch.cuda.cudart(), array.ctypes.data
    # Portable: page-locked for every device, whichever is current.
    if int(cudart.cudaHostRegister(at, array.nbytes, 1)) == 0:
        weakref.finalize(array, cudart.cudaHostUnregister, at)


def payload_tensor(header: Header) -> torch.Tensor:
    """The payload as a tensor; it must not be written through."""
    with warnings.catch_warnings():
        # torch warns that the buffer is read-only.
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(np.frombuffer(header.payload, np.uint8))


class RecordCodec:
    def __init__(
        self,
        codec: str,
        signature: str,
        *,
        chunks: int,
        chunk_bytes: int,
    ):
        if codec not in CODECS:
            raise ValueError(f"unknown codec {codec}")
        self.codec = codec
        self.signature = signature
        self._chunks = chunks
        self._chunk = chunk_bytes

    def step(self, size: int) -> int:
        # Assembling or staging a record costs one GIL hand-off per chunk;
        # while the worker thread is busy each waits up to the switch
        # interval, so the chunk count must not grow with the record size.
        return max(self._chunk, -(-size // self._chunks))

    def encode(
        self,
        key: bytes,
        data: memoryview,
        raw_bytes: int,
        checksum: list[int],
        frames: tuple[list[int], int] = ([], 0),
    ) -> tuple[memoryview, int, int]:
        """The read-only envelope, its payload size and header size.
        ``data`` is the flat bytes, or for ANS their encoding in
        ``frames``: frame lengths and raw bytes per frame."""
        header = {
            "key": key.hex(),
            "signature": self.signature,
            "codec": self.codec,
            "raw_bytes": raw_bytes,
            "checksum": checksum,
        }
        if frames[0]:
            header["frames"], header["chunk_bytes"] = frames
        meta = json.dumps(header, sort_keys=True).encode()
        prefix = MAGIC + struct.pack("<I", len(meta)) + meta
        envelope = np.empty(len(prefix) + len(data), dtype=np.uint8)
        envelope[: len(prefix)] = np.frombuffer(prefix, dtype=np.uint8)
        np.copyto(envelope[len(prefix) :], np.frombuffer(data, np.uint8))
        envelope.flags.writeable = False
        return memoryview(envelope), len(data), len(meta)

    def parse(self, key: bytes, envelope: memoryview, expected: int) -> Header:
        if bytes(envelope[: len(MAGIC)]) != MAGIC:
            raise ValueError("bad magic")
        (size,) = struct.unpack_from("<I", envelope, len(MAGIC))
        start = len(MAGIC) + 4
        header = json.loads(bytes(envelope[start : start + size]))
        if (
            header["key"] != key.hex()
            or header["signature"] != self.signature
            or header["codec"] != self.codec
            or header["raw_bytes"] != expected
        ):
            raise ValueError("record identity mismatch")
        parsed = Header(
            raw_bytes=expected,
            checksum=header["checksum"],
            meta_bytes=size,
            payload=envelope[start + size :],
            frames=header.get("frames", []),
            chunk_bytes=header.get("chunk_bytes", 0),
        )
        if sum(parsed.frames) != (len(parsed.payload) if parsed.frames else 0):
            raise ValueError("frame lengths mismatch")
        if self.codec == "IDENTITY":
            if parsed.frames or len(parsed.payload) != expected:
                raise ValueError("payload size mismatch")
        elif expected and (
            parsed.chunk_bytes <= 0
            or len(parsed.frames) != -(-expected // parsed.chunk_bytes)
        ):
            raise ValueError("frame count mismatch")
        return parsed

    def stage(
        self, header: Header, out: np.ndarray, pool: Executor
    ) -> tuple[list[Future], int]:
        """Start copying the payload into ``out`` on ``pool``: one job per
        ``step`` bytes, in order, and ``step``."""
        step = self.step(len(out))
        return [
            pool.submit(
                np.copyto,
                out[at : at + step],
                np.frombuffer(header.payload[at : at + step], dtype=np.uint8),
            )
            for at in range(0, len(out), step)
        ], step
