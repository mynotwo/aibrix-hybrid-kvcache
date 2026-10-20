# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from aibrix_kvcache.integration.vllm.hybrid.logic import (
    PARTS,
    GroupPlan,
    HybridIndex,
    HybridPlanError,
    record_key,
)

# Mirrors 141: groups 0-3 Mamba align, 4 ring (skipped), 5 attention.
PLAN = GroupPlan(
    kinds={0: "mamba", 1: "mamba", 4: "skip", 5: "attention"},
    block_sizes={0: 400, 1: 400, 4: 4, 5: 400},
    align=400,
    hash_block_size=400,
)


def hashes(n):
    return [bytes([i]) * 4 for i in range(n)]


def test_record_key_binds_namespace_hash_boundary_part_start_and_rank():
    base = record_key("ns", b"h", 800, rank=0, part="span", start=400)
    assert base == record_key("ns", b"h", 800, rank=0, part="span", start=400)
    for other in (
        record_key("ns2", b"h", 800, rank=0, part="span", start=400),
        record_key("ns", b"g", 800, rank=0, part="span", start=400),
        record_key("ns", b"h", 400, rank=0, part="span", start=400),
        record_key("ns", b"h", 800, rank=1, part="span", start=400),
        record_key("ns", b"h", 800, rank=0, part="state", start=400),
        record_key("ns", b"h", 800, rank=0, part="span", start=0),
    ):
        assert other != base


def test_span_covers_only_its_attention_blocks():
    block_ids = ([0, 0, 11], [0, 0, 12], [30], [], [40], [5, 6, 7, 8])
    assert PLAN.span_ids(block_ids, start=400, end=1200) == {
        0: [],
        1: [],
        5: [6, 7],
    }
    assert PLAN.span_ids(block_ids, start=0, end=1200)[5] == [5, 6, 7]


def test_state_is_only_the_boundary_mamba_block():
    block_ids = ([0, 0, 11], [0, 0, 12], [30], [], [40], [5, 6, 7, 8])
    assert PLAN.load_state_ids(block_ids, boundary=1200) == {
        0: [11],
        1: [12],
        5: [],
    }
    offloads = [(0, 103, 800), (1, 106, 800), (0, 101, 400)]
    assert PLAN.save_state_ids(offloads, boundary=800) == {
        0: [103],
        1: [106],
        5: [],
    }


def test_save_state_requires_every_mamba_group_at_the_boundary():
    with pytest.raises(HybridPlanError, match="group 1"):
        PLAN.save_state_ids([(0, 103, 800)], boundary=800)


def test_plan_rejects_unallocated_unaligned_or_empty_targets():
    with pytest.raises(HybridPlanError, match="aligned"):
        PLAN.load_state_ids(([1], [2], [], [], [3], [4]), boundary=500)
    with pytest.raises(HybridPlanError, match="group 5"):
        PLAN.span_ids(([0, 1], [0, 2], [], [], [3], [0, 4]), start=0, end=800)
    with pytest.raises(HybridPlanError, match="span"):
        PLAN.span_ids(([1], [2], [], [], [3], [4, 5]), start=800, end=800)
    with pytest.raises(HybridPlanError, match="span"):
        PLAN.span_ids(([1], [2], [], [], [3], [4, 5]), start=200, end=800)


def test_save_boundary_is_largest_common_offer_within_prefill():
    offloads = [
        (0, 1, 400),
        (1, 2, 400),
        (0, 3, 800),
        (1, 4, 800),
        (0, 5, 1200),
    ]
    assert PLAN.save_boundary(offloads, prefill_end=1000) == 800
    assert PLAN.save_boundary(offloads, prefill_end=399) is None
    assert PLAN.save_boundary([], prefill_end=1000) is None


def test_save_boundary_ignores_unaligned_partial_tail_offers():
    # vLLM 0.30 also offers the CoW copy of a partial tail at its exact,
    # unaligned token count; it must not hide the aligned boundary.
    offloads = [(0, 1, 400), (1, 2, 400), (0, 3, 650), (1, 4, 650)]
    assert PLAN.save_boundary(offloads, prefill_end=1000) == 400


def test_plan_rejects_unknown_kind_and_misaligned_block_size():
    with pytest.raises(HybridPlanError, match="kind"):
        GroupPlan({0: "sliding"}, {0: 400}, align=400, hash_block_size=400)
    with pytest.raises(HybridPlanError, match="divide"):
        GroupPlan({0: "attention"}, {0: 300}, align=400, hash_block_size=400)
    with pytest.raises(HybridPlanError, match="hash"):
        GroupPlan({0: "attention"}, {0: 400}, align=400, hash_block_size=300)


def _store(index, block_hash, boundary, parent=None, ranks=(0,), parts=PARTS):
    if not index.saving(block_hash, boundary):
        index.begin_save(block_hash, boundary, parent)
    for rank in ranks:
        for part in parts:
            index.mark_stored(
                part,
                block_hash,
                boundary,
                rank=rank,
                put=f"{part}{rank}",
                start=parent[1] if parent and part == "span" else 0,
            )


def test_index_hits_only_when_every_rank_stored_and_leaves_one_token():
    index = HybridIndex(world_size=2)
    request = hashes(4)  # 1600 tokens of hashes
    _store(index, request[1], 800, ranks=(0,))
    assert index.longest_hit(request, PLAN, num_tokens=1601) == 0
    _store(index, request[1], 800, ranks=(1,))
    assert index.longest_hit(request, PLAN, num_tokens=1601) == 800
    # Boundary must leave at least one token to compute.
    assert index.longest_hit(request[:2], PLAN, num_tokens=800) == 0


def test_a_record_needs_its_state_and_every_span_down_to_zero():
    index = HybridIndex(world_size=1)
    request = hashes(4)
    _store(index, request[1], 800)
    parent = index.parent_for(request, PLAN, boundary=1600)
    assert parent == (request[1], 800)
    _store(index, request[3], 1600, parent)
    assert index.chain(request[3], 1600) == [
        (request[3], 800, 1600),
        (request[1], 0, 800),
    ]
    assert index.longest_hit(request, PLAN, num_tokens=1601) == 1600
    assert index.puts(request[3], 1600) == ("state0", "span0", "span0")

    # Losing the parent's span loses the child; the parent's state alone
    # keeps nothing restorable either.
    index.mark_failed("span", request[1], 800)
    assert not index.has(request[3], 1600) and not index.has(request[1], 800)
    assert index.longest_hit(request, PLAN, num_tokens=1601) == 0


def test_a_missing_state_hides_only_its_own_boundary():
    index = HybridIndex(world_size=1)
    request = hashes(4)
    _store(index, request[1], 800)
    _store(index, request[3], 1600, (request[1], 800), parts=("span",))
    assert not index.has(request[3], 1600)
    assert index.longest_hit(request, PLAN, num_tokens=1601) == 800


def test_parent_must_share_the_prefix():
    index = HybridIndex(world_size=1)
    _store(index, b"a" * 4, 400)
    other = [b"b" * 4, b"c" * 4, b"d" * 4]
    assert index.parent_for(other, PLAN, boundary=1200) is None
    # A span being saved is already a parent for the next turn.
    index.begin_save(other[1], 800)
    assert index.parent_for(other, PLAN, boundary=1200) == (other[1], 800)


def test_pending_save_defers_lookup_until_every_rank_reports():
    index = HybridIndex(world_size=2)
    request = hashes(3)
    index.begin_save(request[1], 800)
    assert index.saving(request[1], 800)
    # A request whose prefix covers a record still being written must wait.
    assert index.pending_for(request, PLAN, num_tokens=1201)
    assert not index.pending_for(hashes(1) + [b"z" * 4], PLAN, num_tokens=1201)

    _store(index, request[1], 800, ranks=(0,))
    assert index.pending_for(request, PLAN, num_tokens=1201)
    index.mark_stored("state", request[1], 800, rank=1)
    assert index.saving(request[1], 800)
    index.mark_stored("span", request[1], 800, rank=1)
    assert not index.saving(request[1], 800)
    assert not index.pending_for(request, PLAN, num_tokens=1201)
    assert index.longest_hit(request, PLAN, num_tokens=1201) == 800


def test_failed_save_clears_pending_without_indexing():
    index = HybridIndex(world_size=2)
    index.begin_save(b"h", 800)
    index.mark_stored("state", b"h", 800, rank=0)
    index.save_failed("span", b"h", 800)
    assert not index.saving(b"h", 800)
    assert not index.has(b"h", 800)
    assert index.parent_for([b"x", b"h"], PLAN, boundary=1200) is None


def test_a_re_save_forgets_parts_of_the_old_span():
    index = HybridIndex(world_size=1)
    request = hashes(3)
    _store(index, request[1], 800, parts=("state",))
    index.save_failed("span", request[1], 800)
    # The state survived, but the next save at 800 starts from scratch.
    index.begin_save(request[1], 800, None)
    index.mark_stored("span", request[1], 800, rank=0)
    assert not index.has(request[1], 800)


def test_a_save_after_an_ancestor_span_failed_starts_from_zero():
    index = HybridIndex(world_size=1)
    request = hashes(4)
    _store(index, request[0], 400)
    parent = index.parent_for(request, PLAN, boundary=800)
    _store(index, request[1], 800, parent)
    # The oldest span is gone (e.g. evicted): 800's chain is broken.
    index.mark_failed("span", request[0], 400)
    assert not index.has(request[1], 800)
    assert index.parent_for(request, PLAN, boundary=1200) is None
    _store(index, request[2], 1200)
    assert index.chain(request[2], 1200) == [(request[2], 0, 1200)]


def test_a_late_report_of_an_earlier_save_attempt_is_ignored():
    index = HybridIndex(world_size=2)
    request = hashes(3)
    _store(index, request[0], 400, ranks=(0, 1))
    # First attempt at 800 had no parent (start 0); it failed on rank 1 ...
    index.begin_save(request[1], 800, None)
    index.save_failed("state", request[1], 800)
    # ... and the re-save builds on 400.
    index.begin_save(request[1], 800, (request[0], 400))
    index.mark_stored("state", request[1], 800, rank=0)
    index.mark_stored("state", request[1], 800, rank=1)
    # The first attempt's rank-1 span [0, 800) lands late.
    index.mark_stored("span", request[1], 800, rank=1, start=0)
    index.save_failed("span", request[1], 800, start=0)
    index.mark_stored("span", request[1], 800, rank=0, start=400)
    assert not index.has(request[1], 800)
    assert index.saving(request[1], 800)
    index.mark_stored("span", request[1], 800, rank=1, start=400)
    assert index.has(request[1], 800)
