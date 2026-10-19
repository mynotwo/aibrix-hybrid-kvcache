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

import base64
import hashlib
import importlib.util
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aibrix_kvcache.compression import (
    CacheCompressionAdapter,
    CompressionIdentityError,
    CompressionIntegrityError,
    IdentityCompressionAdapter,
    ZstdCompressionAdapter,
)
from aibrix_kvcache.l2.connectors.shfs import SHFSConnector

# Frozen before Task 3 using the Phase 0 adapter at commit 4282280.
# Unlike a runtime-generated fixture, this catches persisted-format changes.
LEGACY_PHASE0_ENVELOPE = base64.b64decode(
    "QUlCWEtWQzEcAQAAeyJibG9ja19rZXkiOiJwcmVmaXgvbGVnYWN5LXBoYXNlMCIs"
    "ImNhY2hlX2dyb3VwIjoiZnVsbF9hdHRlbnRpb24iLCJjaGVja3N1bSI6IjI3NTIz"
    "ZTNmZDVkMWFmOTQ4MTNjOTZjODk1OTU1OGJjNjQxYjMzMzQ4NjRiZTE1MzdiMWVh"
    "MDBmMWNiMjc1YzkiLCJjb2RlYyI6InpzdGQiLCJjb2RlY192ZXJzaW9uIjoxLCJj"
    "b21wcmVzc2VkX2xlbmd0aCI6MzQsImR0eXBlIjoiYmZsb2F0MTYiLCJnZW5lcmF0"
    "aW9uIjo3LCJsYXlvdXQiOiJMQ05EIiwicmF3X2xlbmd0aCI6MjcyLCJzaGFwZSI6"
    "WzE2LDE2XX0otS/9YBAAxQAAiHBoYXNlMC16c3RkLWNhY2hlAQD5RM0l"
)
BLOCK_IDENTITY = dict(
    shape=(16, 128),
    dtype="bfloat16",
    layout="LCND",
    block_key="prefix/缓存",
    generation=9,
    cache_group="full_attention",
)


@pytest.mark.parametrize(
    "adapter_cls", [IdentityCompressionAdapter, ZstdCompressionAdapter]
)
@pytest.mark.parametrize("raw", [b"", b"kv" * 1024], ids=["empty", "kv"])
def test_envelope_accounting_closes(adapter_cls, raw):
    adapter: CacheCompressionAdapter = adapter_cls()
    block = adapter.encode(raw, **BLOCK_IDENTITY)
    envelope = adapter.serialize(block)
    sizes = adapter.encoded_sizes(block, envelope)

    assert sizes.payload == len(block.payload)
    assert sizes.metadata == struct.unpack_from("<I", envelope, 8)[0]
    assert sizes.framing == 12
    assert sizes.payload + sizes.metadata + sizes.framing == sizes.envelope
    assert sizes.envelope == len(envelope)
    assert adapter.decode(adapter.deserialize(envelope)) == raw


@pytest.mark.parametrize(
    "adapter_cls", [IdentityCompressionAdapter, ZstdCompressionAdapter]
)
def test_envelope_accounting_rejects_inconsistent_envelope_size(adapter_cls):
    adapter = adapter_cls()
    block = adapter.encode(b"kv", **BLOCK_IDENTITY)
    envelope = adapter.serialize(block)

    with pytest.raises(ValueError):
        adapter.encoded_sizes(block, envelope[:-1])
    with pytest.raises(ValueError):
        adapter.deserialize(envelope[:-1])


@pytest.mark.parametrize(
    "adapter_cls", [IdentityCompressionAdapter, ZstdCompressionAdapter]
)
def test_trusted_accounting_uses_no_json_or_payload_copy(
    adapter_cls, monkeypatch
):
    adapter = adapter_cls()
    block = adapter.encode(b"kv" * (1024 * 1024), **BLOCK_IDENTITY)

    class NoSlice(bytes):
        def __getitem__(self, item):
            if isinstance(item, slice):
                raise AssertionError("accounting copied envelope bytes")
            return super().__getitem__(item)

    envelope = NoSlice(adapter.serialize(block))

    def forbidden(*args, **kwargs):
        raise AssertionError("trusted accounting performed JSON work")

    monkeypatch.setattr(
        "aibrix_kvcache.compression.adapter.json.loads", forbidden
    )
    monkeypatch.setattr(
        "aibrix_kvcache.compression.adapter.json.dumps", forbidden
    )
    sizes = adapter.encoded_sizes(block, envelope, trusted=True)
    assert sizes.payload == len(block.payload)
    assert sum(sizes[:3]) == len(envelope)


@pytest.mark.parametrize(
    "invalid", [b"", b"AIBXKVC1", b"AIBXKVC1\xff\xff\xff\xff"]
)
def test_accounting_rejects_truncated_or_oversized_framing(invalid):
    adapter = IdentityCompressionAdapter()
    block = adapter.encode(b"kv", **BLOCK_IDENTITY)
    for trusted in (False, True):
        with pytest.raises(ValueError):
            adapter.encoded_sizes(block, invalid, trusted=trusted)


@pytest.mark.parametrize(
    "adapter_cls", [IdentityCompressionAdapter, ZstdCompressionAdapter]
)
@pytest.mark.parametrize(
    "corruption", ["magic", "metadata", "payload", "generation"]
)
def test_envelope_accounting_rejects_malformed_or_mismatched_block(
    adapter_cls, corruption
):
    adapter = adapter_cls()
    block = adapter.encode(b"kv" * 1024, **BLOCK_IDENTITY)
    envelope = adapter.serialize(block)
    if corruption == "magic":
        invalid_envelope = b"!" + envelope[1:]
    elif corruption == "metadata":
        invalid_envelope = envelope[:12] + b"!" + envelope[13:]
    elif corruption == "payload":
        invalid_envelope = envelope[:-1] + bytes([envelope[-1] ^ 1])
    else:
        different_block = block._replace(
            metadata=block.metadata._replace(generation=8)
        )
        invalid_envelope = adapter.serialize(different_block)

    assert len(invalid_envelope) == len(envelope)
    with pytest.raises(ValueError):
        adapter.encoded_sizes(block, invalid_envelope)


def test_identity_codec_preserves_payload_without_compression():
    raw = bytes(range(256)) * 8
    block = IdentityCompressionAdapter().encode(raw, **BLOCK_IDENTITY)

    assert block.payload == raw
    assert block.metadata.codec == "identity"
    assert block.metadata.codec_version == 1
    assert block.metadata.raw_length == block.metadata.compressed_length == 2048
    assert block.metadata.checksum == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_key", "prefix/other"),
        ("generation", 8),
        ("cache_group", "other"),
        ("shape", (32, 128)),
        ("dtype", "float32"),
        ("layout", "NCLD"),
    ],
)
def test_identity_codec_rejects_mismatched_identity(field, value):
    adapter = IdentityCompressionAdapter()
    block = adapter.encode(b"kv", **BLOCK_IDENTITY)
    expected = {
        f"expected_{key}": item for key, item in BLOCK_IDENTITY.items()
    }
    assert adapter.decode(block, **expected) == b"kv"
    expected[f"expected_{field}"] = value
    with pytest.raises(CompressionIdentityError):
        adapter.decode(block, **expected)


@pytest.mark.parametrize("metadata", [{"codec": "zstd"}, {"codec_version": 2}])
def test_identity_codec_rejects_unsupported_codec_or_version(metadata):
    adapter = IdentityCompressionAdapter()
    block = adapter.encode(b"kv", **BLOCK_IDENTITY)
    with pytest.raises(CompressionIdentityError):
        adapter.decode(block._replace(metadata=block.metadata._replace(**metadata)))


@pytest.mark.parametrize("payload", [b"k", b"kw"])
def test_identity_codec_rejects_corrupted_length_or_checksum(payload):
    adapter = IdentityCompressionAdapter()
    block = adapter.encode(b"kv", **BLOCK_IDENTITY)
    with pytest.raises(CompressionIntegrityError):
        adapter.decode(block._replace(payload=payload))


def test_legacy_phase0_zstd_envelope_remains_readable_and_serializable():
    adapter = ZstdCompressionAdapter()
    block = adapter.deserialize(LEGACY_PHASE0_ENVELOPE)

    assert block.metadata.codec == "zstd"
    assert block.metadata.codec_version == 1
    assert block.metadata.block_key == "prefix/legacy-phase0"
    assert block.metadata.generation == 7
    assert block.metadata.cache_group == "full_attention"
    assert block.metadata.shape == (16, 16)
    assert block.metadata.dtype == "bfloat16"
    assert block.metadata.layout == "LCND"
    assert adapter.decode(block) == b"phase0-zstd-cache" * 16
    assert adapter.serialize(block) == LEGACY_PHASE0_ENVELOPE
    assert adapter.encoded_sizes(block, LEGACY_PHASE0_ENVELOPE) == (
        34, 284, 12, 330
    )


def load_adapter_module():
    path = (
        Path(__file__).parents[1]
        / "aibrix_kvcache"
        / "compression"
        / "adapter.py"
    )
    spec = importlib.util.spec_from_file_location("compression_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompressionAdapterTest(unittest.TestCase):
    def test_zstd_block_round_trip_preserves_bytes_and_identity(self):
        module = load_adapter_module()
        raw = bytes(range(64)) * 32
        adapter = module.ZstdCompressionAdapter()

        encoded = adapter.encode(
            raw,
            shape=(2, 2, 16, 4, 128),
            dtype="bfloat16",
            layout="LCND",
            block_key="prefix/0001",
            generation=7,
            cache_group="full_attention",
        )

        self.assertEqual(encoded.metadata.raw_length, 2048)
        self.assertEqual(
            encoded.metadata.compressed_length, len(encoded.payload)
        )
        self.assertEqual(encoded.metadata.shape, (2, 2, 16, 4, 128))
        self.assertEqual(encoded.metadata.dtype, "bfloat16")
        self.assertEqual(encoded.metadata.layout, "LCND")
        self.assertEqual(encoded.metadata.block_key, "prefix/0001")
        self.assertEqual(encoded.metadata.generation, 7)
        self.assertEqual(encoded.metadata.cache_group, "full_attention")
        self.assertEqual(encoded.metadata.codec, "zstd")
        self.assertEqual(encoded.metadata.codec_version, 1)
        self.assertEqual(adapter.decode(encoded), raw)

    def test_serialized_envelope_restores_variable_payload_and_metadata(self):
        module = load_adapter_module()
        adapter = module.ZstdCompressionAdapter()
        encoded = adapter.encode(
            b"a" * 4096,
            shape=(32, 128),
            dtype="bfloat16",
            layout="NCLD",
            block_key="prefix/variable",
            generation=3,
            cache_group="full_attention",
        )

        envelope = adapter.serialize(encoded)
        restored = adapter.deserialize(envelope)

        self.assertLess(len(restored.payload), restored.metadata.raw_length)
        self.assertEqual(restored, encoded)
        self.assertEqual(adapter.decode(restored), b"a" * 4096)

    def test_corrupted_payload_is_rejected_without_returning_bytes(self):
        module = load_adapter_module()
        adapter = module.ZstdCompressionAdapter()
        encoded = adapter.encode(
            bytes(range(256)) * 8,
            shape=(16, 128),
            dtype="bfloat16",
            layout="LCND",
            block_key="prefix/corrupt",
            generation=1,
            cache_group="full_attention",
        )
        corrupted = module.CompressedBlock(
            payload=encoded.payload[:-1] + bytes([encoded.payload[-1] ^ 1]),
            metadata=encoded.metadata,
        )

        with self.assertRaises(module.CompressionIntegrityError):
            adapter.decode(corrupted)

    def test_restore_rejects_wrong_block_identity_or_generation(self):
        module = load_adapter_module()
        adapter = module.ZstdCompressionAdapter()
        encoded = adapter.encode(
            b"kv" * 1024,
            shape=(16, 128),
            dtype="bfloat16",
            layout="LCND",
            block_key="prefix/current",
            generation=9,
            cache_group="full_attention",
        )

        with self.assertRaises(module.CompressionIdentityError):
            adapter.decode(
                encoded,
                expected_block_key="prefix/other",
                expected_generation=9,
                expected_cache_group="full_attention",
            )

        with self.assertRaises(module.CompressionIdentityError):
            adapter.decode(
                encoded,
                expected_block_key="prefix/current",
                expected_generation=9,
                expected_cache_group="full_attention",
                expected_shape=(32, 128),
            )

        unsupported = encoded._replace(
            metadata=encoded.metadata._replace(codec_version=2)
        )
        with self.assertRaises(module.CompressionIdentityError):
            adapter.decode(unsupported)
        with self.assertRaises(module.CompressionIdentityError):
            adapter.decode(
                encoded,
                expected_block_key="prefix/current",
                expected_generation=8,
                expected_cache_group="full_attention",
            )


if __name__ == "__main__":
    unittest.main()


@pytest.mark.asyncio
async def test_shfs_variable_payload_round_trip_uses_actual_length(tmp_path):
    executor = ThreadPoolExecutor(max_workers=1)
    connector = SHFSConnector(str(tmp_path), executor)
    try:
        assert connector.open().is_ok()
        assert connector.feature.variable_payload

        put_status = await connector.put_bytes(b"block-key", b"abc" * 17)
        get_status = await connector.get_bytes(b"block-key")

        assert put_status.is_ok()
        assert get_status.is_ok()
        assert get_status.get() == b"abc" * 17
        assert connector._key_to_filepath(b"block-key").stat().st_size == 51
    finally:
        connector.close()
        executor.shutdown(wait=True)
