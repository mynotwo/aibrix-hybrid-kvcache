# AIBrix Hybrid KV Cache

**What must a hybrid-model prefix cache preserve before it can be reused?**

A hybrid prefix combines recurrent state at an aligned boundary with
full-attention spans. This downstream AIBrix project studies how to represent,
store and restore those pieces together, and how to inspect the compression
work around them.

The directly runnable entry is a variable-length byte path with bounded MEMORY
storage, lossless envelopes and trace accounting. Whole-model output equivalence
remains unresolved. This is an independent experimental integration, not an
official AIBrix release or a validated serving-speed demonstration.

## Three parts you can inspect independently

| Part | Entry point | Evidence and boundary |
|---|---|---|
| Storage contract | [CPU quickstart](tools/CPU_QUICKSTART.md) | Actual MEMORY operations, byte equality, capacity/lease guards and trace events on synthetic inputs |
| Hybrid state/span design | [Technical article](tools/HYBRID_KV_ARTICLE.md) | Record design, accounting and published failure evidence |
| Whole-model restore | [Correctness investigation](https://github.com/mynotwo/aibrix-hybrid-kvcache/issues/1) | Strict model acceptance remains unresolved; tensor round trips do not establish it |

Start with the CPU example to inspect the contract without model weights or a
cluster. Use the investigation to judge the remaining integration work.

## Repository layout

| Path | Contents |
|---|---|
| `aibrix_kvcache/integration/vllm/hybrid` | vLLM external connector for hybrid models: record layout, save/restore logic, byte store, GPU codec |
| `aibrix_kvcache/compression` | Lossless self-describing envelopes (IDENTITY, zstd) |
| `aibrix_kvcache/compression_profiler` | Stage-level trace recorder and byte accounting |
| `aibrix_kvcache/l1`, `l2`, `memory`, `transport` | AIBrix KV cache core: CPU cache, L2 backends (MEMORY byte API added here), allocators |
| `tools/` | CPU demo, compression-impact analysis, Qwen3-Next benchmark scripts |
| `tests/` | CPU test suite; CUDA/nvCOMP cases skip without hardware |
| `csrc/`, `CMakeLists.txt` | Optional CUDA kernels built only for CUDA targets |

See [REDESIGN.md](REDESIGN.md) for why the repository has this shape.

## Explore

- [Connector guide](aibrix_kvcache/integration/vllm/hybrid/README.md):
  configuration, failure behavior, limitations and benchmark commands.
- [Technical note](tools/qwen_next_hybrid/TECHNICAL_NOTE.md):
  architecture, what compression can improve, and evidence boundaries.
- [Verification record](tools/qwen_next_hybrid/VERIFICATION.md):
  exact source version, environments, test commands and unresolved acceptance.

The implementation lives at the repository root: `compression` provides
lossless envelopes, `compression_profiler` records causal stages and byte
accounting, and `integration/vllm/hybrid` implements the external connector.

## Try the CPU example

Run the real variable-length MEMORY byte path, lossless envelopes and trace
recorder without a GPU, model weights, or Kubernetes. The [CPU quickstart](tools/CPU_QUICKSTART.md)
includes a pinned checkout, requirements, expected output and troubleshooting.
It produces readable JSON/JSONL artifacts and fails if byte equality, identity,
checksum, capacity or reader protection does not hold.

After installing the guide's core dependencies, run from the checkout root:

```bash
PYTHONPATH=. python -m tools.cpu_quickstart \
  --output work/cpu-demo-1
```

A successful run reports byte-identity, checksum, capacity and reader-protection
checks and writes a summary plus trace events. The named state/span payloads are
synthetic byte vectors, not restored model state.

[Read the technical article](tools/HYBRID_KV_ARTICLE.md)
for the hybrid state/span design, published synthetic data and current
whole-model failure evidence. This is a research entry point, not a validated
model-serving or speedup demonstration.

## Run the CPU checks

Use Python 3.11 or 3.12. No GPU, vLLM install or cluster is needed:

```bash
git clone https://github.com/mynotwo/aibrix-hybrid-kvcache.git
cd aibrix-hybrid-kvcache
python -m venv .venv
. .venv/bin/activate
make install        # requirements/test.txt + ruff
make test-hybrid    # hybrid connector, byte store and codec tests (seconds)
make test           # full CPU suite (a few minutes)
make lint
make demo           # CPU storage-contract demo into work/cpu-demo
```

CUDA/nvCOMP tests skip when their hardware or optional dependency is absent.
These tests use constructed tensors; they do not establish model compatibility.
The current CUDA checks run with vLLM 0.30.0, Torch 2.13.0+cu130 and CUDA 13.0.
Other versions require independent verification; `>=0.30` is not a support promise.

## Upstream collaboration

- [Design RFC](https://github.com/vllm-project/aibrix/issues/3220)
- [First backend PR (draft)](https://github.com/vllm-project/aibrix/pull/3221)
- [Model correctness investigation](https://github.com/mynotwo/aibrix-hybrid-kvcache/issues/1)

## Contribute

Start with an issue that includes the source commit, runtime/model revisions,
hardware, configuration, and a minimal reproducer. Keep original traces private
if they contain request data or environment details. Upstream-facing work is
split into small independently tested changes; this repository keeps the full
experimental integration available for review.

Licensed under Apache-2.0. The KV cache core is derived from
[AIBrix](https://github.com/vllm-project/aibrix); original copyright headers
and contribution history are preserved. This repository does not ship the
AIBrix Kubernetes control plane; use upstream AIBrix for that.
