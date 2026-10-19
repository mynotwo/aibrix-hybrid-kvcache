import pytest
import torch

from aibrix_kvcache.integration.vllm.hybrid.layout import (
    HybridLayout,
    HybridLayoutError,
)

NUM_BLOCKS = 7


def _caches():
    return {
        # Mamba-style layer: tuple of state tensors, blocks first.
        "linear.0": (
            torch.arange(NUM_BLOCKS * 6, dtype=torch.float32).reshape(
                NUM_BLOCKS, 2, 3
            ),
            torch.arange(NUM_BLOCKS * 4, dtype=torch.bfloat16).reshape(
                NUM_BLOCKS, 4
            ),
        ),
        # Attention layer with a leading K/V dimension.
        "attn.0": torch.arange(
            2 * NUM_BLOCKS * 3, dtype=torch.bfloat16
        ).reshape(2, NUM_BLOCKS, 3),
        "ring.0": torch.zeros(NUM_BLOCKS, 4),
    }


GROUPS = {
    0: {"linear.0": "mamba"},
    4: {"ring.0": "skip"},
    5: {"attn.0": "attention"},
}


def test_layout_skips_ring_and_orders_layers_by_group_and_name():
    layout = HybridLayout.build(GROUPS, _caches(), num_blocks=NUM_BLOCKS)

    assert layout.group_ids == (0, 5)
    assert [(s.group_id, s.name, s.tensor_index) for s in layout.slots] == [
        (0, "linear.0", 0),
        (0, "linear.0", 1),
        (5, "attn.0", 0),
    ]
    assert [s.block_dim for s in layout.slots] == [0, 0, 1]


def test_gather_then_scatter_restores_bytes_exactly():
    caches = _caches()
    layout = HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS)
    block_ids = {0: [3], 5: [1, 4]}
    expected = layout.gather(block_ids)
    original = {
        "linear.0": tuple(t.clone() for t in caches["linear.0"]),
        "attn.0": caches["attn.0"].clone(),
    }
    for tensor in (*caches["linear.0"], caches["attn.0"]):
        tensor.zero_()

    layout.scatter(block_ids, expected)

    assert torch.equal(caches["linear.0"][0][3], original["linear.0"][0][3])
    assert torch.equal(caches["linear.0"][1][3], original["linear.0"][1][3])
    assert torch.equal(
        caches["attn.0"][:, [1, 4]], original["attn.0"][:, [1, 4]]
    )
    assert torch.count_nonzero(caches["attn.0"][:, [0, 2, 3, 5, 6]]) == 0


def test_scatter_validates_every_segment_before_writing():
    caches = _caches()
    layout = HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS)
    block_ids = {0: [3], 5: [1, 4]}
    values = layout.gather(block_ids)
    before = caches["linear.0"][0].clone()

    with pytest.raises(HybridLayoutError, match="segment 2 size"):
        layout.scatter(block_ids, [*values[:2], values[2][:-1]])
    assert torch.equal(caches["linear.0"][0], before)


def test_signature_changes_with_shape_dtype_or_group():
    base = HybridLayout.build(GROUPS, _caches(), num_blocks=NUM_BLOCKS)
    caches = _caches()
    caches["attn.0"] = caches["attn.0"].float()
    assert (
        HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS).signature()
        != base.signature()
    )
    moved = {0: GROUPS[0], 4: GROUPS[4], 6: GROUPS[5]}
    assert (
        HybridLayout.build(moved, _caches(), num_blocks=NUM_BLOCKS).signature()
        != base.signature()
    )
    assert (
        HybridLayout.build(GROUPS, _caches(), num_blocks=NUM_BLOCKS).signature()
        == base.signature()
    )


@pytest.mark.parametrize(
    "groups, caches, reason",
    [
        ({0: {"x": "sliding"}}, {"x": torch.zeros(NUM_BLOCKS)}, "unsupported"),
        ({0: {"missing": "attention"}}, {}, "no KV tensor"),
        ({0: {"x": "attention"}}, {"x": torch.zeros(3, 4)}, "block dimension"),
        (
            {0: {"x": "attention"}},
            {"x": torch.zeros(NUM_BLOCKS, NUM_BLOCKS)},
            "block dimension",
        ),
        (
            {0: {"x": "attention", "y": "mamba"}},
            {"x": torch.zeros(NUM_BLOCKS), "y": torch.zeros(NUM_BLOCKS)},
            "mixes",
        ),
        ({0: {"x": "skip"}}, {"x": torch.zeros(NUM_BLOCKS)}, "nothing"),
    ],
)
def test_layout_fails_closed(groups, caches, reason):
    with pytest.raises(HybridLayoutError, match=reason):
        HybridLayout.build(groups, caches, num_blocks=NUM_BLOCKS)


def test_gather_rejects_unknown_or_missing_group_blocks():
    layout = HybridLayout.build(GROUPS, _caches(), num_blocks=NUM_BLOCKS)
    with pytest.raises(HybridLayoutError, match="block ids"):
        layout.gather({0: [1]})
    with pytest.raises(HybridLayoutError, match="out of range"):
        layout.gather({0: [NUM_BLOCKS], 5: [0]})


def test_kernel_blocks_are_grouped_into_manager_blocks_like_vllm():
    # vLLM may split each manager block into ``ratio`` kernel blocks on
    # dim 0 (vllm.v1.kv_cache_interface.group_kernel_blocks).
    ratio = 3
    kernel = torch.arange(NUM_BLOCKS * ratio * 2, dtype=torch.float32)
    caches = {"attn.k": kernel.reshape(NUM_BLOCKS * ratio, 2)}
    layout = HybridLayout.build(
        {5: {"attn.k": "attention"}}, caches, num_blocks=NUM_BLOCKS
    )
    (segment,) = layout.gather({5: [2]})
    expected = caches["attn.k"][2 * ratio : 3 * ratio]
    assert segment == expected.numpy().tobytes()

    layout.scatter({5: [4]}, [segment])
    assert torch.equal(caches["attn.k"][4 * ratio : 5 * ratio], expected)


def test_flat_gather_scatter_round_trip_matches_segments():
    caches = _caches()
    layout = HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS)
    source = {0: [3], 5: [1, 4]}
    flat = layout.gather_flat(source)
    assert flat.dtype == torch.uint8 and flat.dim() == 1
    assert flat.numpy().tobytes() == b"".join(layout.gather(source))

    target = {0: [6], 5: [0, 2]}
    layout.scatter_flat(target, flat)
    assert layout.gather(target) == layout.gather(source)


def test_scatter_flat_rejects_wrong_size_without_writing():
    caches = _caches()
    layout = HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS)
    flat = layout.gather_flat({0: [3], 5: [1, 4]})
    before = caches["attn.0"].clone()
    with pytest.raises(HybridLayoutError, match="flat record size"):
        layout.scatter_flat({0: [6], 5: [0, 2]}, flat[:-1])
    assert torch.equal(caches["attn.0"], before)


def test_gather_into_fills_host_buffer_like_gather_flat():
    caches = _caches()
    layout = HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS)
    source = {0: [3], 5: [1, 4]}
    out = torch.zeros(sum(layout.gather_sizes(source)), dtype=torch.uint8)
    layout.gather_into(source, out)  # CPU tensors: copies are immediate
    assert torch.equal(out, layout.gather_flat(source))
    with pytest.raises(HybridLayoutError, match="flat record size"):
        layout.gather_into(source, out[:-1])


def test_gather_into_and_scatter_flat_copy_scattered_runs_of_blocks():
    caches = _caches()
    layout = HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS)
    # linear.0 (2 tensors, blocks on dim 0) is copied block by block;
    # attn.0 has blocks on dim 1 and goes through index_select/index_copy_.
    source, target = {0: [1, 2, 5], 5: [0]}, {0: [6, 0, 4], 5: [3]}
    out = torch.zeros(sum(layout.gather_sizes(source)), dtype=torch.uint8)
    layout.gather_into(source, out)
    assert torch.equal(out, layout.gather_flat(source))
    layout.scatter_flat(target, out)
    assert layout.gather(target) == layout.gather(source)


@pytest.mark.parametrize("empty", [0, 5])
def test_a_group_without_blocks_contributes_no_bytes(empty):
    caches = _caches()
    layout = HybridLayout.build(GROUPS, caches, num_blocks=NUM_BLOCKS)
    part = {g: ([] if g == empty else [1, 3]) for g in layout.group_ids}
    sizes = layout.gather_sizes(part)
    assert [
        size for slot, size in zip(layout.slots, sizes)
        if slot.group_id == empty
    ] == [0] * sum(slot.group_id == empty for slot in layout.slots)
    out = torch.empty(sum(sizes), dtype=torch.uint8)
    layout.gather_into(part, out)
    assert torch.equal(out, layout.gather_flat(part))

    # Writing the part into other blocks leaves the empty group untouched.
    target = {g: ([] if g == empty else [2, 4]) for g in layout.group_ids}
    before = layout.gather_flat({g: list(range(NUM_BLOCKS))
                                 for g in layout.group_ids})
    layout.scatter_flat(target, out)
    assert torch.equal(layout.gather_flat(target), out)
    after = layout.gather_flat({g: list(range(NUM_BLOCKS))
                                for g in layout.group_ids})
    changed = (before != after).nonzero().flatten()
    assert changed.numel() > 0
    full = layout.gather_sizes({g: list(range(NUM_BLOCKS))
                                for g in layout.group_ids})
    at, untouched = 0, []
    for slot, size in zip(layout.slots, full):
        if slot.group_id == empty:
            untouched.append((at, at + size))
        at += size
    for lo, hi in untouched:
        assert torch.equal(before[lo:hi], after[lo:hi])
