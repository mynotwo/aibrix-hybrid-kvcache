# CPU quickstart: bytes in, verified bytes out

This example runs the actual MEMORY connector, lossless compression adapters,
and trace recorder. It uses public, repetitive synthetic bytes named `state`
and `span`. It does not start vLLM, reconstruct hybrid tensors, or load a model.

## Requirements and fit

| Path | Requirements | What it answers |
| --- | --- | --- |
| This example | Linux or macOS, Python 3.11/3.12, the existing core dependencies | Can I store variable-sized envelopes, restore bytes, and inspect events? |
| GPU record checks | NVIDIA CUDA GPU and optional nvCOMP runtime; see the connector guide | Do constructed tensor records round-trip on that environment? |
| Whole-model serving | Matched engine/model versions and the strict acceptance protocol | Not established by this example; model equivalence remains unresolved |

No GPU, model download, credentials, Redis server, Kubernetes cluster, or
compiled AIBrix extension is needed for the example. Torch is still imported
by the package. Its runtime and installed dependencies consume memory beyond
the demo's 32 KiB payload capacity. Native Windows is not supported by this
command's Unix-only recorder/dependency path.

The checked local environment is macOS arm64, Python 3.12.13, Torch 2.14.1,
and zstandard 0.25.0. Linux/Python 3.11 and 3.12 package CI is a separate check;
read its run result before treating that environment as verified for this tag.

## Run from a new checkout

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

`cpu-quickstart-v1` pins this example, not a production/model-support release.
Python 3.11 also belongs to the package CI matrix. Use `python3.11` instead of
`python3.12` if that is your installed interpreter.

The dependency installation can be much larger than the example itself because
it includes Torch. On Linux, you can install the CPU Torch wheel first using
the [official PyTorch installer](https://pytorch.org/get-started/locally/), then
install the existing core requirements above. Do not install the whole package
with `pip install .` for this source-only example: that starts its native build.

## Expected result

A successful run exits 0 and prints:

```text
PASS identity: 2 records, 12288 raw bytes, 12836 stored bytes, 8 trace events
PASS zstd: 2 records, 12288 raw bytes, 838 stored bytes, 8 trace events
PASS capacity, duplicate-write, lease, identity and checksum guards
Artifacts: work/cpu-demo-1
```

The stored sizes above are observations from the checked environment. Codec or
metadata versions may change sizes. The invariants are exact restored bytes,
valid traces with zero drops, correct byte accounting, and rejected unsafe
operations. A failed invariant exits nonzero; the script has no mock-backend
or synthetic-success fallback.

The output contains:

- `summary.json`: environment, byte counts, ratios and guard results.
- `identity/events.jsonl` and `zstd/events.jsonl`: eight real measured operation
  events per arm, with dependencies for encode → store → load → decode.
- Separate recorder ownership/status sidecars in each arm's directory: run
  validity and dropped-event evidence are retained for both arms.

Read the result without extra packages:

```bash
python -m json.tool work/cpu-demo-1/summary.json
```

These are example-instrumented CPU operations using the real recorder. They
are not complete serving-request traces, do not measure PCIe traffic, and do
not support a TTFT or throughput conclusion. The state/span names do not make
these byte vectors valid recurrent-state or attention layouts.

## What is exercised

1. Store and retrieve two different-length envelopes in a real 32 KiB MEMORY
   connector; compare restored bytes with every original byte.
2. Account for encoded payload, metadata and framing. `stored_bytes` equals
   their sum; `raw_to_envelope_ratio` uses the full stored envelope denominator.
3. Reject a wrong generation and corrupted checksum for each record.
4. In a separate 12-byte cache, reject conflicting duplicates and a 9-byte
   admission while a 4-byte entry is leased; preserve all existing entries.
5. Deny close during that lease, then admit an 8-byte record after release
   and evict eligible LRU entries.

MEMORY capacity counts stored payload bytes, excluding Python objects,
allocator overhead, temporary buffers and Torch. It is process-local and
volatile; it is not cross-process sharing or a total host-memory limit.

## Troubleshooting

| Symptom | Cause or next check |
| --- | --- |
| `No module named aibrix_kvcache` or `tools.cpu_quickstart` | Run from the repository root with the shown `PYTHONPATH`. Check that you checked out `cpu-quickstart-v1`, not the older experimental tag. |
| Missing Torch, zstandard, or another import | Check `python -m pip --version` uses the activated environment, then install the existing core requirements. |
| `--output already exists` | Choose a new path, such as `work/cpu-demo-2`. Existing results are never overwritten. |
| `Failed to read commit hash` / missing `_version` warning | Expected for a source-only checkout; it is not a failed byte-round-trip check. |
| Installation fails with a certificate or proxy error | Check that environment's package-index/TLS access; do not disable certificate verification or change cache logic. |
| Native dependency wheel unavailable | Use the documented Python/platform combination; a native compiler may be needed for existing dependencies such as netifaces. |
| Trace validation, checksum, capacity, or byte comparison fails | Keep the output and stderr, record `git rev-parse HEAD` and `python --version`, and file a minimal issue. Do not relabel it as a pass. |

## Regression checks and evidence

Install test tools only if running the regression suite:

```bash
python -m pip install pytest pytest-asyncio pytest-mock
PYTHONPATH=python/aibrix_kvcache python -m pytest -q \
  python/aibrix_kvcache/tests/test_cpu_quickstart.py \
  python/aibrix_kvcache/tests/test_memory_connector.py \
  python/aibrix_kvcache/tests/test_connector_inheritance.py \
  python/aibrix_kvcache/tests/test_l2cache.py
```

[Published sample data](quickstart-evidence.json) includes the summary,
all 16 trace records, dependency versions and a SHA-256 of the demo source.
Only synthetic inputs are included. For full integration limits, see the
[connector guide](../aibrix_kvcache/integration/vllm/hybrid/README.md),
[upstream backend PR](https://github.com/vllm-project/aibrix/pull/3221), and
[model investigation](https://github.com/mynotwo/aibrix-hybrid-kvcache/issues/1).
