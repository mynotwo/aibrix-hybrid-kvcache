# Hybrid KV cache needs a storage contract before a speedup claim

A hybrid model can need two different kinds of state to resume a prefix:
recurrent state at a valid boundary, and the attention history that boundary
depends on. Moving one buffer successfully is useful evidence. It is not yet
evidence that a model resumed correctly.

I built [AIBrix Hybrid KV Cache](https://github.com/mynotwo/aibrix-hybrid-kvcache)
to investigate that gap. It is an independent experimental extension of
[AIBrix](https://github.com/vllm-project/aibrix). The useful starting point is
small: store variable-sized records, preserve readers during eviction, and
account for the bytes actually stored. A CPU example makes those contracts
inspectable without downloading a model or provisioning GPUs.

## Why a prefix is more than an attention buffer

[vLLM's hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
describes cache management across different attention/state requirements.
For this project's hybrid connector, a reusable boundary combines recurrent
state with a chain of incremental full-attention spans. Both must belong to
the requested record identity. Every required tensor-parallel rank must report
its required parts before the scheduler advertises that boundary as available.

This is a property of the experimental connector, not a claim that ordinary
KV offloading is universally incorrect or that every hybrid architecture has
the same layout. The implementation and its engine/model versions matter.

```mermaid
flowchart TD
    A[Recurrent state at aligned boundary] --> C[Per-rank logical prefix]
    B[Required attention span chain] --> C
    C --> D[Self-describing variable-sized records]
    D --> E[Backend storage and reader leases]
    E --> F[All required ranks and parts reported]
    F --> G[Eligible external prefix]
    G --> H[Restore, validate identity and bytes, write destination]
    H --> I[Separate whole-model output acceptance]
```

A byte-store example exercises the middle of this diagram. It does not run the
scheduler, establish rank completeness, or demonstrate the final model gate.

## The first upstream contribution is a byte-store contract

[PR #3221](https://github.com/vllm-project/aibrix/pull/3221) proposes an optional
`variable_payload` capability, `get_bytes`/`put_bytes`, and a bounded MEMORY
backend. Backends without the capability retain their existing contract.
The full connector, codecs and profiler remain downstream.

The backend's capacity counts stored payload bytes. Reads update LRU order;
eviction skips leased entries. Admission first finds enough eligible capacity,
then evicts the selected batch. If admission cannot succeed, the existing
records remain intact. A duplicate key with identical bytes is idempotent;
different bytes under that key are rejected.

A lease also protects against deletion and close. Typed/shaped memoryviews
count their byte length rather than their first dimension. Contiguous read-only
views can be borrowed under an explicit backing-buffer ownership contract;
read-only does not mean the underlying owner can never mutate the buffer.
Writable or strided inputs are copied.

MEMORY is process-local and volatile. Its byte capacity excludes Python
objects, allocator overhead and transient staging buffers. A 10 GiB setting
is neither a 10 GiB total process-memory limit nor a distributed cache.

## Run a small example before running a model

Use Linux or macOS with Python 3.11/3.12 and the existing package dependencies.
No GPU, vLLM server, Redis server, Kubernetes cluster, model weights or request
credentials are needed. Torch is an existing import dependency even on CPU.

```bash
git clone --branch cpu-quickstart-v1 --depth 1 \
  https://github.com/mynotwo/aibrix-hybrid-kvcache.git
cd aibrix-hybrid-kvcache
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r python/aibrix_kvcache/requirements/core.txt
PYTHONPATH=python/aibrix_kvcache python -m tools.cpu_quickstart \
  --output work/cpu-demo-1
```

The [full quickstart](CPU_QUICKSTART.md) includes platform requirements,
expected output, regression commands and troubleshooting. The pinned tag
identifies the example; it does not announce model or production support.

The script sends two synthetic byte vectors through the real connector:
4,096 bytes named `state` and 8,192 bytes named `span`. Their names do not
make them valid hybrid tensors. Each arm encodes, stores, retrieves, decodes
and checks every byte. Separate checks exercise duplicate writes, protected
admission, safe rejection, close during a lease and subsequent eviction.
It also rejects a wrong generation and a corrupted checksum for each record.

## Count the envelope, not just the compressed payload

These are observations from macOS arm64, Python 3.12.13, Torch 2.14.1 and
zstandard 0.25.0, using deliberately repetitive synthetic input:

| Arm | Raw bytes | Encoded payload | Metadata | Framing | Full stored bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDENTITY | 12,288 | 12,288 | 524 | 24 | 12,836 |
| CPU zstd | 12,288 | 301 | 513 | 24 | 838 |

For both rows, payload + metadata + framing equals full stored bytes.
IDENTITY grows because it still carries a self-describing envelope. A tiny
payload can compress well while its metadata becomes most of the stored
object. Comparing raw bytes only with the compressed payload would hide that
cost. The summary's `raw_to_envelope_ratio` uses full stored bytes as its
denominator.

The [published JSON evidence](quickstart-evidence.json) contains the summary,
all 16 JSONL-derived event records, dependency versions and the demo source
SHA-256. There are eight measured events per arm: four stages for each of two
records. The real recorder's run validity, dependency structure and zero-drop
status are checked. A nonzero exit signals a failed invariant.

This trace is explicit instrumentation of the example's CPU operations. It is
not a complete serving trace or an automatic observation of every inference
stage. Absolute monotonic timestamps are local to the recorded run.

## Stored-byte savings are not a latency result

This example's zstd encoder and decoder run on the CPU. It does not show
compressed bytes traversing PCIe. The experimental GPU path uses a separate,
optional ANS record codec; its codec placement and transfer behavior need their
own measurements.

For a path that really transfers `S_wire` bytes at effective bandwidth `B`,
transfer time is bounded below by `S_wire / B`. Stored-object size is not a
substitute for `S_wire`. Encoding, decoding, destination writes, waiting and
remaining model computation can dominate the request. Overlapping stages must
be evaluated on the actual timeline rather than added as if they were serial.
For a required multi-rank restore, the request cannot finish before its slowest
required rank is ready.

The binding question is therefore whether saved critical-path transfer work
exceeds added codec and staging work after all dependencies are respected.
A smaller stored object alone cannot answer it. This article makes no TTFT,
throughput or end-to-end speedup claim.

## Byte equality has not settled model correctness

The [model investigation](https://github.com/mynotwo/aibrix-hybrid-kvcache/issues/1#issuecomment-6300253903)
records both passing and failing trials under the same strict acceptance gate.
At source `a00a2620`, an IDENTITY trial passed, a repeat failed, and an ANS trial
passed. Per-rank byte verification and destination readback matched for the
required state/span parts in all three trials. The failing repeat still failed
whole-model output acceptance. The
[sanitized model evidence](https://github.com/mynotwo/aibrix-hybrid-kvcache/releases/download/hybrid-v0.1.1-experimental/model-proof-summary.json)
is public; private model inputs and raw request logs are not.

A connector-disabled control also observed output variability, but its output
variants did not explain the failing connector trial. That is a reason to
investigate matched baselines, not a reason to excuse the failure. The cause
remains unresolved. Passing individual runs and record checks do not establish
stable whole-model equivalence, and the acceptance criterion has not been
relaxed to turn the failure into a pass.

## A reviewable path upstream

The [RFC](https://github.com/vllm-project/aibrix/issues/3220) covers the broader
design. The first PR stays limited to the optional byte API and MEMORY backend.
It remains a Draft for interface feedback; upstream execution also depends on
maintainer workflow approval. Public downstream CI evidence is linked from the
PR, with its exact source version rather than an implied result for all future
changes.

Useful first feedback is concrete: can a consumer use the optional byte API
without changing fixed-block backends; do lease and duplicate-key semantics fit
the existing connector interface; and does the byte-capacity contract make
memory accounting clear enough? A reproduction report should include the
source revision, platform/dependency versions, exact command, nonzero exit or
observed invariant failure, and public synthetic artifacts where possible.

The project is ready for that bounded evaluation. Broader model compatibility
and serving-performance claims require additional evidence.
