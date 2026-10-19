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
"""Scheduler-side rules for incremental hybrid records (no vLLM imports).

A save at prefix boundary ``b`` stores two parts: ``state`` (every Mamba
group's state at ``b``) and ``span`` (every attention group's blocks for
tokens ``[p, b)``, ``p`` being the largest boundary already saved for the
same prefix, else 0). Restoring ``b`` reads its state and the spans down to
0, so a multi-turn conversation never rewrites its prefix. vLLM's Mamba
"align" mode keeps the state after ``(i + 1) * block_size`` tokens in block
``i`` of a Mamba group.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

KINDS = ("attention", "mamba", "skip")
PARTS = ("state", "span")


class HybridPlanError(ValueError):
    """A boundary cannot be mapped to blocks safely."""


def record_key(
    namespace: str,
    block_hash: bytes,
    boundary: int,
    *,
    rank: int,
    part: str,
    start: int = 0,
):
    digest = hashlib.sha256()
    for item in (
        namespace.encode(),
        block_hash,
        str(boundary).encode(),
        part.encode(),
        str(start).encode(),
    ):
        digest.update(len(item).to_bytes(8, "little"))
        digest.update(item)
    digest.update(rank.to_bytes(4, "little"))
    return digest.digest()


@dataclass(frozen=True)
class Part:
    """One stored object of a record: its identity and its blocks."""

    part: str
    block_hash: bytes
    end: int
    start: int
    block_ids: Mapping[int, Sequence[int]]


@dataclass(frozen=True)
class GroupPlan:
    kinds: Mapping[int, str]
    block_sizes: Mapping[int, int]
    align: int
    hash_block_size: int

    def __post_init__(self):
        for group_id, kind in self.kinds.items():
            if kind not in KINDS:
                raise HybridPlanError(f"group {group_id}: unknown kind {kind}")
            if kind != "skip" and self.align % self.block_sizes[group_id]:
                raise HybridPlanError(
                    f"group {group_id}: block size must divide {self.align}"
                )
        if self.align % self.hash_block_size:
            raise HybridPlanError("hash block size must divide alignment")

    def groups(self, kind: str) -> list[int]:
        return sorted(g for g, k in self.kinds.items() if k == kind)

    def _check_boundary(self, boundary: int) -> None:
        if boundary <= 0 or boundary % self.align:
            raise HybridPlanError(
                f"boundary {boundary} is not aligned to {self.align}"
            )

    def _empty(self, kind: str) -> dict[int, list[int]]:
        return {group_id: [] for group_id in self.groups(kind)}

    def span_ids(
        self, block_ids: Sequence[Sequence[int]], *, start: int, end: int
    ) -> dict[int, list[int]]:
        """Attention blocks covering tokens ``[start, end)``."""
        self._check_boundary(end)
        if start and start % self.align or not 0 <= start < end:
            raise HybridPlanError(f"bad span [{start}, {end})")
        result = self._empty("mamba")
        for group_id in self.groups("attention"):
            size = self.block_sizes[group_id]
            ids = list(block_ids[group_id][start // size : end // size])
            if len(ids) != (end - start) // size or 0 in ids:
                raise HybridPlanError(
                    f"group {group_id}: real blocks for [{start}, {end}) "
                    f"needed, got {ids}"
                )
            result[group_id] = ids
        return dict(sorted(result.items()))

    def load_state_ids(
        self, block_ids: Sequence[Sequence[int]], *, boundary: int
    ) -> dict[int, list[int]]:
        """Destination Mamba blocks for the state at ``boundary``."""
        self._check_boundary(boundary)
        result = self._empty("attention")
        for group_id in self.groups("mamba"):
            index = boundary // self.block_sizes[group_id] - 1
            ids = block_ids[group_id]
            if index >= len(ids) or ids[index] == 0:
                raise HybridPlanError(
                    f"group {group_id}: no block allocated at {boundary}"
                )
            result[group_id] = [ids[index]]
        return dict(sorted(result.items()))

    def save_state_ids(
        self, offloads: Iterable[tuple[int, int, int]], *, boundary: int
    ) -> dict[int, list[int]]:
        """Source Mamba blocks: the exact boundary-state offloads."""
        self._check_boundary(boundary)
        exact = {
            group_id: block_id
            for group_id, block_id, tokens in offloads
            if tokens == boundary
        }
        result = self._empty("attention")
        for group_id in self.groups("mamba"):
            if group_id not in exact:
                raise HybridPlanError(
                    f"group {group_id}: no boundary state at {boundary}"
                )
            result[group_id] = [exact[group_id]]
        return dict(sorted(result.items()))

    def load_parts(
        self, block_ids, block_hash: bytes, boundary: int, chain
    ) -> list[Part]:
        """Destination blocks of a stored record's state and spans."""
        state = self.load_state_ids(block_ids, boundary=boundary)
        return [Part("state", block_hash, boundary, 0, state)] + [
            Part(
                "span",
                span_hash,
                end,
                start,
                self.span_ids(block_ids, start=start, end=end),
            )
            for span_hash, start, end in chain
        ]

    def save_boundary(
        self, offloads: Iterable[tuple[int, int, int]], *, prefill_end: int
    ) -> Optional[int]:
        """Largest aligned boundary offered for every Mamba group in the prompt.

        vLLM also offers a partial tail's CoW copy at its unaligned token
        count; a record is only addressable at aligned boundaries.
        """
        offered = defaultdict(set)
        for group_id, _, tokens in offloads:
            if tokens <= prefill_end and not tokens % self.align:
                offered[tokens].add(group_id)
        mamba = set(self.groups("mamba"))
        return max(
            (b for b, groups in offered.items() if mamba <= groups),
            default=None,
        )

    def save_parts(
        self, block_ids, offloads, block_hash: bytes, boundary: int, *,
        start: int
    ) -> list[Part]:
        """Source blocks of the state at ``boundary`` and its new span."""
        return [
            Part(
                "state",
                block_hash,
                boundary,
                0,
                self.save_state_ids(offloads, boundary=boundary),
            ),
            Part(
                "span",
                block_hash,
                boundary,
                start,
                self.span_ids(block_ids, start=start, end=boundary),
            ),
        ]


@dataclass
class HybridIndex:
    """Which record parts every rank has stored, as reported by workers."""

    world_size: int
    # (part, block hash, boundary) -> ranks that stored it.
    _ranks: dict = field(default_factory=lambda: defaultdict(set))
    # (block hash, boundary) -> its span's parent (hash, boundary) or None.
    _parents: dict = field(default_factory=dict)
    # Records whose save was scheduled but not yet stored on every rank.
    _saving: set = field(default_factory=set)
    # Profiler operation ids of each part's per-rank puts.
    _puts: dict = field(default_factory=lambda: defaultdict(list))

    def begin_save(
        self,
        block_hash: bytes,
        boundary: int,
        parent: Optional[tuple[bytes, int]] = None,
    ) -> None:
        # A re-save may pick another parent, so forget the old parts.
        for part in PARTS:
            self._ranks.pop((part, block_hash, boundary), None)
            self._puts.pop((part, block_hash, boundary), None)
        self._saving.add((block_hash, boundary))
        self._parents[(block_hash, boundary)] = parent

    def saving(self, block_hash: bytes, boundary: int) -> bool:
        return (block_hash, boundary) in self._saving

    def _complete(self, part: str, block_hash: bytes, boundary: int) -> bool:
        ranks = self._ranks.get((part, block_hash, boundary))
        return ranks is not None and len(ranks) == self.world_size

    def mark_stored(
        self,
        part: str,
        block_hash: bytes,
        boundary: int,
        *,
        rank: int,
        start: int = 0,
        put: str = "",
    ):
        if not self._current(part, block_hash, boundary, start):
            return
        self._ranks[(part, block_hash, boundary)].add(rank)
        if put:
            self._puts[(part, block_hash, boundary)].append(put)
        if all(self._complete(p, block_hash, boundary) for p in PARTS):
            self._saving.discard((block_hash, boundary))

    def _current(self, part, block_hash, boundary, start) -> bool:
        """Whether a worker report is of the save now indexed at ``boundary``.

        A re-save may pick another parent; a late report of the earlier
        attempt's span (another object) must not count for this one.
        """
        key = (block_hash, boundary)
        if part == "state":
            return start == 0
        return key in self._parents and start == (
            self._parents[key][1] if self._parents[key] else 0
        )

    def mark_failed(
        self, part: str, block_hash: bytes, boundary: int, *, start: int = 0
    ) -> None:
        if not self._current(part, block_hash, boundary, start):
            return
        self._ranks.pop((part, block_hash, boundary), None)
        self._puts.pop((part, block_hash, boundary), None)
        if part == "span":
            # Nothing may build on a span that is gone.
            self._parents.pop((block_hash, boundary), None)

    def save_failed(
        self, part: str, block_hash: bytes, boundary: int, *, start: int = 0
    ) -> None:
        if not self._current(part, block_hash, boundary, start):
            return
        self._saving.discard((block_hash, boundary))
        self.mark_failed(part, block_hash, boundary, start=start)

    def chain(
        self, block_hash: bytes, boundary: int
    ) -> Optional[list[tuple[bytes, int, int]]]:
        """``(hash, start, end)`` of every span restoring ``boundary``.

        None unless the state at ``boundary`` and every span down to token 0
        are stored on every rank.
        """
        if not self._complete("state", block_hash, boundary):
            return None
        spans, key = [], (block_hash, boundary)
        while key is not None:
            if key not in self._parents or not self._complete("span", *key):
                return None
            parent = self._parents[key]
            spans.append((key[0], parent[1] if parent else 0, key[1]))
            key = parent
        return spans

    def has(self, block_hash: bytes, boundary: int) -> bool:
        return self.chain(block_hash, boundary) is not None

    def puts(self, block_hash: bytes, boundary: int) -> tuple[str, ...]:
        """Put operations of every part a restore of ``boundary`` reads."""
        found = list(self._puts.get(("state", block_hash, boundary), ()))
        for span_hash, _, end in self.chain(block_hash, boundary) or ():
            found += self._puts.get(("span", span_hash, end), ())
        return tuple(found)

    def _boundaries(self, block_hashes, plan, num_tokens):
        boundary = (num_tokens - 1) // plan.align * plan.align
        while boundary > 0:
            index = boundary // plan.hash_block_size - 1
            if index < len(block_hashes):
                yield block_hashes[index], boundary
            boundary -= plan.align

    def parent_for(
        self, block_hashes: Sequence[bytes], plan: GroupPlan, *, boundary: int
    ) -> Optional[tuple[bytes, int]]:
        """Largest earlier boundary of this prefix with a span saved or
        being saved: a new span at ``boundary`` only has to cover the rest.
        """
        for key in self._boundaries(block_hashes, plan, boundary):
            if self._alive(key):
                return key
        return None

    def _alive(self, key) -> bool:
        """Every span from ``key`` down to 0 is stored or still saving."""
        while key is not None:
            if key not in self._parents or not (
                key in self._saving or self._complete("span", *key)
            ):
                return False
            key = self._parents[key]
        return True

    def pending_for(
        self, block_hashes: Sequence[bytes], plan: GroupPlan, *, num_tokens: int
    ) -> bool:
        """True if a record this request could load is still being saved."""
        return any(
            key in self._saving
            for key in self._boundaries(block_hashes, plan, num_tokens)
        )

    def longest_hit(
        self, block_hashes: Sequence[bytes], plan: GroupPlan, *, num_tokens: int
    ) -> int:
        """Largest stored boundary that leaves at least one token to run."""
        for block_hash, boundary in self._boundaries(
            block_hashes, plan, num_tokens
        ):
            if self.has(block_hash, boundary):
                return boundary
        return 0
