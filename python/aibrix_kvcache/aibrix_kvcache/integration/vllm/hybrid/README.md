# Hybrid KV cache offloading for vLLM

`AIBrixHybridOffloadingConnector` saves and restores prefix caches of hybrid
models that combine recurrent state with full-attention layers, through
AIBrix L2 backends using the external connector API targeted at vLLM 0.30.0.
Other runtime versions and model layouts are not certified.

A record is the Mamba state at an aligned boundary plus the attention
blocks before it. A request whose prefix is no longer in the GPU prefix
cache gets it restored instead of recomputed.

## Evidence boundary

This is an experimental integration, not a validated model-support claim.
Constructed-tensor CPU and H800/CUDA tests pass, including ANS bytewise
round trips and corruption checks. Whole-model acceptance remains unresolved:
the historical run used for the 256K observation reports FAIL (missing
per-rank verification and generated-token mismatch). The cause of the
output difference has not been established. See the
[verification record](../../../../tools/qwen_next_hybrid/VERIFICATION.md).

## Performance interpretation

Historical development runs motivated this implementation, but mixed source
versions and incomplete whole-model acceptance prevent a release performance
claim. No speedup guarantee is attached to this source snapshot. Use the
benchmark below to collect matched-version evidence for your environment.

Compare recomputation, IDENTITY restore and ANS restore separately. External
reuse can avoid prefill work; compression can reduce stored/transferred bytes.
A shorter restore window alone does not establish lower end-to-end TTFT.
See the [technical note](../../../../tools/qwen_next_hybrid/TECHNICAL_NOTE.md).

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
- `IDENTITY` stores the raw bytes. CPU zstd belongs to the general L2
  compression adapter and is not offered in this hybrid path.

Deployment:

- The development launcher uses `--numa-bind` and container `SYS_NICE` so
  worker host memory can follow each GPU's NUMA node. Validate the effect
  on your topology rather than assuming a fixed bandwidth improvement.
- Set per-rank MEMORY capacity through
  `AIBRIX_KV_CACHE_OL_MEMORY_CAPACITY_BYTES`; budget from the actual record
  sizes and number of retained prefixes for your layout.

## Failure behaviour

The connector requests recomputation when record restore fails. The tested
record-level cases below do not prove engine recovery from every failure:

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
  The cause has not been established; numerical differences and connector
  correctness both need investigation. This blocks an output-equivalence claim.
- Background saves still consume host memory, CPU and copy bandwidth. A
  historical run stalled during host allocation/pinning; its cause remains
  unknown. Dropping busy saves limits queueing but does not establish that
  all resource-pressure failures preserve engine responsiveness.

## Benchmark

`tools/qwen_next_hybrid/bench.sh` serves each arm, times TTFT by prompt
length (`ttft_lengths.py`), then checks the profiler trace
(`tools/qwen_next_acceptance.py`) and breaks restores into stages
(`tools/compression_impact_analyze.py`). Connector arms enable local-hit
verification. This diagnostic mode is for acceptance evidence; use a separate
matched experiment to measure production-mode overhead. The strict gate can
fail on unresolved whole-model differences and returns nonzero in that case:

```
MODEL=/models/Qwen3-Next SRC=$PWD/python/aibrix_kvcache DEPS=/path/to/nvcomp \
RUNS=/data/runs tools/qwen_next_hybrid/bench.sh "8150 32150 128150" vanilla identity ans
```
