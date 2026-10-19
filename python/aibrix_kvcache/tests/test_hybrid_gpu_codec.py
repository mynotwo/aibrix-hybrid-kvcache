import asyncio
import importlib.util
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from aibrix_kvcache.integration.vllm.hybrid import gpu_codec as codec_module
from aibrix_kvcache.integration.vllm.hybrid.gpu_codec import checksum
from aibrix_kvcache.integration.vllm.hybrid.layout import HybridLayout
from aibrix_kvcache.integration.vllm.hybrid.store import (
    HybridStore,
    HybridStoreError,
)
from aibrix_kvcache.l2.connectors.connector import Connector, ConnectorConfig

gpu_codec = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("nvidia.nvcomp") is None,
    reason="needs CUDA and nvidia-nvcomp",
)


def _bytes(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=generator)


def test_checksum_detects_a_changed_byte_and_reordered_rows():
    flat = _bytes(3 * 4096 + 7)
    reference = checksum(flat).tolist()
    changed = flat.clone()
    changed[-1] ^= 1
    swapped = torch.cat([flat[4096:8192], flat[:4096], flat[8192:]])
    assert checksum(changed).tolist() != reference
    assert checksum(swapped).tolist() != reference


def test_checksum_does_not_depend_on_how_many_rows_are_summed_at_once(
    monkeypatch,
):
    flat = _bytes(7 * 4096 + 3)
    whole = checksum(flat).tolist()
    monkeypatch.setattr(codec_module, "_SLICE", 2)  # bounded scratch
    assert checksum(flat).tolist() == whole


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_and_cpu_checksums_agree():
    flat = _bytes(5 * 4096 + 2)
    assert checksum(flat.cuda()).tolist() == checksum(flat).tolist()


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("AIBRIX_KV_CACHE_OL_MEMORY_CAPACITY_BYTES", "100000000")
    connector = Connector.create(
        ConnectorConfig(
            backend_name="MEMORY",
            namespace="gpu-codec-test",
            partition_id="p0",
            executor=ThreadPoolExecutor(1),
        )
    )
    assert connector.open().is_ok()
    yield connector
    connector.close()


def _ans_store(backend, **kwargs):
    torch.manual_seed(0)
    # Interleaved like vLLM's hybrid buffer: one block holds every layer.
    shared = torch.randn(8, 2, 64, 256).to(torch.bfloat16).cuda()
    caches = {"attn": shared[:, 0], "linear": shared[:, 1]}
    groups = {0: {"linear": "mamba"}, 1: {"attn": "attention"}}
    layout = HybridLayout.build(groups, caches, num_blocks=8)
    return HybridStore(layout, backend, "ANS", **kwargs)


@gpu_codec
def test_ans_record_is_encoded_on_the_gpu_and_restored_exactly(backend):
    store = _ans_store(backend)
    source, target = {0: [1], 1: [2, 3, 4]}, {0: [6], 1: [5, 7, 0]}
    original = store.layout.gather(source)
    stats = store.save(b"key", source)
    assert stats["stored_bytes"] < stats["raw_bytes"]
    store.load(b"key", target)
    assert store.layout.gather(target) == original
    store.close()


@gpu_codec
def test_ans_record_is_encoded_frame_by_frame(backend):
    # Copy chunks that end mid-frame: each frame waits for its own bytes.
    store = _ans_store(backend, chunk_bytes=24 << 10)
    store._ans.frame_bytes = 40 << 10  # each frame's scratch is bounded
    source, target = {0: [1], 1: [2, 3, 4]}, {0: [6], 1: [5, 7, 0]}
    original = store.layout.gather(source)
    raw = store.save(b"key", source)["raw_bytes"]
    record = memoryview(asyncio.run(backend.get_bytes(b"key")).value)
    header = store._record.parse(b"key", record, raw)
    assert len(header.frames) == -(-raw // (40 << 10)) > 1
    assert all(n % 256 == 0 for n in header.frames)
    store.load(b"key", target)
    assert store.layout.gather(target) == original
    store.close()


@gpu_codec
def test_ans_save_from_inference_mode_like_vllm(backend):
    store = _ans_store(backend)
    source, target = {0: [1], 1: [2, 3]}, {0: [6], 1: [5, 7]}
    with torch.inference_mode():
        store.save(b"key", source)
        store.load(b"key", target)
    assert store.layout.gather(target) == store.layout.gather(source)
    store.close()


@gpu_codec
def test_corrupted_ans_record_is_rejected_before_blocks_are_written(backend):
    store = _ans_store(backend)
    store.save(b"key", {0: [1], 1: [2, 3]})
    record = bytearray(asyncio.run(backend.get_bytes(b"key")).value)
    record[-100] ^= 0xFF
    assert asyncio.run(backend.delete(b"key")).is_ok()
    assert asyncio.run(backend.put_bytes(b"key", bytes(record))).is_ok()
    target = {0: [6], 1: [5, 7]}
    before = store.layout.gather(target)
    with pytest.raises(HybridStoreError, match="integrity"):
        store.load(b"key", target)
    assert store.layout.gather(target) == before
    store.close()
