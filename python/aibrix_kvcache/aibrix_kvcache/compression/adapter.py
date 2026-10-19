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

import hashlib
import json
import struct
from typing import NamedTuple, Protocol

import zstandard


class CompressionIntegrityError(ValueError):
    """The stored block cannot be restored without data loss."""


class CompressionIdentityError(ValueError):
    """The stored block does not match the requested cache identity."""


class CompressionMetadata(NamedTuple):
    shape: tuple[int, ...]
    dtype: str
    layout: str
    codec: str
    codec_version: int
    raw_length: int
    compressed_length: int
    checksum: str
    block_key: str
    generation: int
    cache_group: str


class CompressedBlock(NamedTuple):
    payload: bytes
    metadata: CompressionMetadata


class EncodedSizes(NamedTuple):
    """Stored envelope bytes, excluding backend protocol or bus traffic."""

    payload: int
    metadata: int
    framing: int
    envelope: int


class CacheCompressionAdapter(Protocol):
    """Lossless cache-block codec with exact persisted-byte accounting."""

    def encode(
        self,
        raw: bytes,
        *,
        shape: tuple[int, ...],
        dtype: str,
        layout: str,
        block_key: str,
        generation: int,
        cache_group: str,
    ) -> CompressedBlock: ...

    def decode(
        self,
        block: CompressedBlock,
        *,
        expected_block_key: str | None = None,
        expected_generation: int | None = None,
        expected_cache_group: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: str | None = None,
        expected_layout: str | None = None,
    ) -> bytes: ...

    def serialize(self, block: CompressedBlock) -> bytes: ...

    def deserialize(self, envelope: bytes) -> CompressedBlock: ...

    def encoded_sizes(
        self, block: CompressedBlock, envelope: bytes, *, trusted: bool = False
    ) -> EncodedSizes: ...


class _EnvelopeCompressionAdapter:
    """Share the Phase 0 metadata, integrity checks and persisted format."""

    _MAGIC = b"AIBXKVC1"
    _codec: str

    def _encode_payload(self, raw: bytes) -> bytes:
        raise NotImplementedError

    def _decode_payload(self, block: CompressedBlock) -> bytes:
        raise NotImplementedError

    def encode(
        self,
        raw: bytes,
        *,
        shape: tuple[int, ...],
        dtype: str,
        layout: str,
        block_key: str,
        generation: int,
        cache_group: str,
    ) -> CompressedBlock:
        payload = self._encode_payload(raw)
        metadata = CompressionMetadata(
            shape=shape,
            dtype=dtype,
            layout=layout,
            codec=self._codec,
            codec_version=1,
            raw_length=len(raw),
            compressed_length=len(payload),
            checksum=hashlib.sha256(raw).hexdigest(),
            block_key=block_key,
            generation=generation,
            cache_group=cache_group,
        )
        return CompressedBlock(payload=payload, metadata=metadata)

    def decode(
        self,
        block: CompressedBlock,
        *,
        expected_block_key: str | None = None,
        expected_generation: int | None = None,
        expected_cache_group: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: str | None = None,
        expected_layout: str | None = None,
    ) -> bytes:
        if (
            block.metadata.codec != self._codec
            or block.metadata.codec_version != 1
        ):
            raise CompressionIdentityError(
                "compressed block codec or version is unsupported"
            )
        expected = (
            ("block_key", expected_block_key),
            ("generation", expected_generation),
            ("cache_group", expected_cache_group),
            ("shape", expected_shape),
            ("dtype", expected_dtype),
            ("layout", expected_layout),
        )
        for field, value in expected:
            if value is not None and getattr(block.metadata, field) != value:
                raise CompressionIdentityError(
                    f"compressed block {field} does not match request"
                )
        raw = self._decode_payload(block)
        if len(raw) != block.metadata.raw_length:
            raise CompressionIntegrityError("decompressed length mismatch")
        if hashlib.sha256(raw).hexdigest() != block.metadata.checksum:
            raise CompressionIntegrityError("decompressed checksum mismatch")
        return raw

    def serialize(self, block: CompressedBlock) -> bytes:
        metadata = block.metadata._asdict()
        metadata["shape"] = list(block.metadata.shape)
        header = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return (
            self._MAGIC
            + struct.pack("<I", len(header))
            + header
            + block.payload
        )

    def deserialize(self, envelope: bytes) -> CompressedBlock:
        if not envelope.startswith(self._MAGIC):
            raise ValueError("invalid compression envelope magic")
        offset = len(self._MAGIC)
        (header_length,) = struct.unpack_from("<I", envelope, offset)
        offset += 4
        metadata_dict = json.loads(
            envelope[offset : offset + header_length].decode("utf-8")
        )
        metadata_dict["shape"] = tuple(metadata_dict["shape"])
        metadata = CompressionMetadata(**metadata_dict)
        payload = envelope[offset + header_length :]
        if len(payload) != metadata.compressed_length:
            raise ValueError("compressed length mismatch")
        return CompressedBlock(payload=payload, metadata=metadata)

    def encoded_sizes(
        self, block: CompressedBlock, envelope: bytes, *, trusted: bool = False
    ) -> EncodedSizes:
        """Count envelope bytes; default validates full block/envelope equality.

        trusted=True is only for an envelope just serialized from this block,
        or a block just deserialized from this exact envelope. It performs no
        JSON work or payload copying; callers must establish that provenance.
        Decode/integrity validation is still required before restoring bytes.
        """
        framing = len(self._MAGIC) + struct.calcsize("<I")
        if len(envelope) < framing or not envelope.startswith(self._MAGIC):
            raise ValueError("invalid compression envelope framing")
        (metadata,) = struct.unpack_from("<I", envelope, len(self._MAGIC))
        payload = len(block.payload)
        if (payload != block.metadata.compressed_length
                or payload + metadata + framing != len(envelope)):
            raise ValueError("envelope size does not match block")
        if not trusted and self.deserialize(envelope) != block:
            raise ValueError("envelope does not match block")
        return EncodedSizes(payload, metadata, framing, len(envelope))


class IdentityCompressionAdapter(_EnvelopeCompressionAdapter):
    """Preserve raw payload bytes with shared metadata and integrity checks."""

    _codec = "identity"

    def _encode_payload(self, raw: bytes) -> bytes:
        return raw

    def _decode_payload(self, block: CompressedBlock) -> bytes:
        return block.payload


class ZstdCompressionAdapter(_EnvelopeCompressionAdapter):
    """Losslessly encode one cache block with self-describing metadata."""

    _codec = "zstd"

    def __init__(self) -> None:
        self._compressor = zstandard.ZstdCompressor(level=1)
        self._decompressor = zstandard.ZstdDecompressor()

    def _encode_payload(self, raw: bytes) -> bytes:
        return self._compressor.compress(raw)

    def _decode_payload(self, block: CompressedBlock) -> bytes:
        try:
            return self._decompressor.decompress(
                block.payload, max_output_size=block.metadata.raw_length
            )
        except zstandard.ZstdError as exc:
            raise CompressionIntegrityError(
                "compressed payload is invalid"
            ) from exc
