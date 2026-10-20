# Verification record — 2026-10-20

## Source and scope

Record and scheduler tests use source commit `b0b4dbc0` (the delivery hybrid
connector plus the CPU stream-context fix). Later launch changes update docs
and benchmark failure handling; they do not change record or scheduler logic.

| Check | Environment | Result |
| --- | --- | --- |
| Hybrid layout, logic, store and codec | macOS, CPU, Python 3.14 | 56 passed; 7 CUDA tests skipped |
| Same four test files | Linux, H800, Python 3.12, Torch 2.13.0+cu130, CUDA 13.0 | 63 passed |
| Compression/profiler, MEMORY, L2 and acceptance unit suites | macOS, CPU | 660 passed |
| Benchmark/acceptance request-ID integration (after launch fix) | macOS, CPU | 17 acceptance tests passed |
| Hybrid source and tests Ruff | macOS | Passed |
| Benchmark failure smoke | Stub executables, no real server | Unknown arm exits 2; readiness timeout exits 1; request failure preserves exit 17; owned container cleanup confirmed |

Constructed-tensor tests are not whole-model tests. The CUDA runtime image
contains vLLM 0.30.0; passing these tests does not certify every connector API
interaction or model supported by that image. CPU Python 3.14 is an additional
local observation, not a package-support commitment.

## CI alignment follow-up

The complete package pre-commit configuration also passes locally, including
Ruff, pinned formatting, codespell, ClangFormat, PyMarkdown, mypy and headers.
The CI follow-up makes tool module names consistent, documents the borrowed
read-only-buffer contract, and adds explicit types to existing dynamic trace
and staging paths. It does not relax the model acceptance gate.

The CI-aligned source was rechecked on H800: all 63 record/scheduler/codec tests
passed again. Public CI/build status is tracked separately from these local
results and must not be inferred from this document.

## Test commands

From the repository root, with the package's dependencies available:

```bash
PYTHONPATH=python/aibrix_kvcache python -m pytest -q \
  python/aibrix_kvcache/tests/test_hybrid_layout.py \
  python/aibrix_kvcache/tests/test_hybrid_logic.py \
  python/aibrix_kvcache/tests/test_hybrid_store.py \
  python/aibrix_kvcache/tests/test_hybrid_gpu_codec.py

PYTHONPATH=python/aibrix_kvcache python -m pytest -q \
  python/aibrix_kvcache/tests/test_compression*.py \
  python/aibrix_kvcache/tests/test_memory_connector.py \
  python/aibrix_kvcache/tests/test_l2cache.py \
  python/aibrix_kvcache/tests/test_qwen_next*.py

ruff check python/aibrix_kvcache/aibrix_kvcache/integration/vllm/hybrid \
  python/aibrix_kvcache/tests/test_hybrid*.py
bash -n python/aibrix_kvcache/tools/qwen_next_hybrid/{serve,bench}.sh
```

## Whole-model acceptance

UNKNOWN / NOT ACCEPTED: the historical development run reports FAIL because
per-rank verification is absent and external-resume output tokens differ from
local-prefix-hit tokens. No cause is established. The strict acceptance gate
is preserved; it has not been relaxed to make the run pass.

A full acceptance result needs the same source/runtime/model versions,
bytewise evidence for every required group and rank, verified local miss and
external restore, a justified model-output criterion, and tested failure
fallback. No release performance headline is derived from that failed run.

## Publication review scope

All 335 downstream blob versions reachable from existing origin branches but
not from their upstream main baseline were scanned for common GitHub-token,
AWS-access-key, private-key, JWT and live-key patterns; no pattern findings
were reported. This is a bounded automated check, not proof that arbitrary
secrets cannot exist. Old development branches contain environment paths;
private datasets, weights and raw request logs are not added by this release.
Original upstream history and Apache-2.0 attribution are retained.
