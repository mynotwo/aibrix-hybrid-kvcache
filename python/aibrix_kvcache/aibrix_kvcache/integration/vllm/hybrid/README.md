# Hybrid KV cache offloading for vLLM

`AIBrixHybridOffloadingConnector` saves and restores prefix caches of hybrid
models, Mamba (linear attention) layers plus full attention layers such as
Qwen3-Next, through AIBrix L2 backends, on unmodified vLLM >= 0.30.

A record is the Mamba state at an aligned boundary plus the attention
blocks before it. A request whose prefix is no longer in the GPU prefix
cache gets it restored instead of recomputed.

## What it saves

Qwen3-Next (80B, TP8, H800), MEMORY backend, `--numa-bind`. Median of 6
prompts per length; the restore window runs from the first record read to
the last block written, over all 8 ranks.

| Prompt | Recompute TTFT | Restore TTFT (ANS) | Restore window IDENTITY / ANS |
|---|---|---|---|
| 8K | 0.21 s | 0.15 s | 6.3 / 6.2 ms |
| 32K | 0.91 s | 0.19 s | 13.8 / 12.0 ms |
| 64K | 1.9 s | 0.24 s | 23.5 / 18.9 ms |
| 128K | 5.0–6.0 s | 0.29 s | 43.8 / 34.5 ms |
| 192K | 8.6–10.9 s | 0.36 s | 63.9 / 49.4 ms |
| 256K | 11–15 s | 0.44 s | 84 / 64 ms |

- The 8K–64K TTFTs come from an earlier build whose restore windows were
  1.4–2.5x longer. The windows in this table are from the current code.
- Each rank holds about 12 KB of record per token: 1.6 GB at 128K and
  3.2 GB at 256K.
- The restore window grows linearly with the prompt. Above 32K it is
  bound by PCIe host-to-device bandwidth (47–55 GB/s per GPU with all 8
  copying).
- ANS cuts the bytes crossing PCIe by 1.34x, so its window is 24% shorter
  at 256K.

## Configuration

```
--kv-transfer-config '{"kv_connector": "AIBrixHybridOffloadingConnector",
  "kv_connector_module_path": "aibrix_kvcache.integration.vllm.hybrid.connector",
  "kv_role": "kv_both", "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {"backend": "MEMORY",
    "compression": "ANS", "namespace": "qwen-next"}}'
```

| Key | Default | Meaning |
|---|---|---|
| `backend` | required | AIBrix L2 backend (`MEMORY` keeps records page-locked in the worker) |
| `compression` | required | `ANS` (recommended) or `IDENTITY` |
| `namespace` | required | key namespace; separate per model and layout |
| `save_wait_s` | 10 | longest a request waits for an in-flight save of its prefix before recomputing |
| `save_enabled` | true | diagnostic: run the connector without saving |
| `verify_local_hits` | false | diagnostic: compare local prefix hits with their records |

Codecs:

- `ANS` runs nvCOMP on the GPU and needs `nvidia-nvcomp-cu13`. Records are
  encoded in 64 MB frames, so the scratch memory stays bounded.
- `IDENTITY` stores the raw bytes. A CPU codec (zstd) is not offered: it
  made every restore slower at every length.

Deployment:

- Pass `--numa-bind` to vLLM, and `--cap-add=SYS_NICE` to its container.
  This keeps each worker's host memory on its GPU's NUMA node. Without it,
  the slowest GPU's host-to-device copy halves (23 vs 47 GB/s).
- MEMORY capacity per rank: about 12 KB × tokens × records to keep, set
  through `AIBRIX_KV_CACHE_OL_MEMORY_CAPACITY_BYTES`.

## Failure behaviour

A record that fails is recomputed. A failure never costs the engine:

- A save that cannot get GPU or host memory, or that the backend refuses,
  is skipped.
- While `max_inflight` (4) saves are still in progress, a new save is
  dropped, not waited for.
- A restore that hits a missing or corrupted record (GPU checksum), a
  layout mismatch, or GPU out of memory falls back to recompute through
  vLLM's `kv_load_failure_policy`.
- A request waits for an in-flight save of its prefix at most
  `save_wait_s`.
- A key that is already stored is not stored again.

## Known limits

- A cold prompt that ends exactly on a Mamba block boundary (a multiple of
  400 tokens for Qwen3-Next) stores no record. vLLM does not offer that
  boundary to the connector.
- A restored request and a local prefix hit may decode different tokens.
  This is vLLM's numerical nondeterminism, which shows up between any two
  prefill paths.
- Saves cost host CPU: per 1.6 GB record, about 0.5 s with ANS and 0.7 s
  with IDENTITY, off the worker thread.
- Once, in one 256K IDENTITY run, a save stalled in host memory allocation
  and pinning for 25 minutes. The cause is unknown and it did not recur.
  Saves are now dropped rather than waited for, so such a stall no longer
  takes the engine down.

## Benchmark

`tools/qwen_next_hybrid/bench.sh` serves each arm, times TTFT by prompt
length (`ttft_lengths.py`), then checks the profiler trace
(`tools/qwen_next_acceptance.py`) and breaks restores into stages
(`tools/compression_impact_analyze.py`):

```
MODEL=/models/Qwen3-Next SRC=$PWD/python/aibrix_kvcache DEPS=/path/to/nvcomp \
RUNS=/data/runs tools/qwen_next_hybrid/bench.sh "8150 32150 128150" vanilla identity ans
```
