# Redesign

## What this project is for

A research library for offloading the prefix cache of hybrid models (recurrent
Mamba-style state plus full-attention KV) and for measuring what lossless
compression actually saves along that path. It answers one question: what must
a stored hybrid prefix contain, and what does storing it cost in bytes and
stages, before anyone claims it is reusable or faster?

It is a downstream of the AIBrix KV cache library, not a fork of the AIBrix
platform.

## What changed

Before: a full copy of vllm-project/aibrix (Go controllers, gateway plugins,
CRDs, Helm/kustomize manifests, console apps, docs site, the `python/aibrix`
runtime, 14 workflow files), about 2,650 tracked files. All original work lived
in `python/aibrix_kvcache` (hybrid connector, compression envelopes, profiler,
MEMORY byte API, tools and tests). Outside it only the README differed from
upstream.

After:

| Decision | Reason |
|---|---|
| Removed the Kubernetes control plane, Go module, apps, docs site, `python/aibrix` runtime and their CI | Unmodified upstream code that the hybrid work never imports. Kept in `main` history and in upstream AIBrix. |
| Promoted `python/aibrix_kvcache/*` to the repository root | One package, one `pyproject.toml`, so `pip install -e .`, `pytest`, and `PYTHONPATH=.` work from the checkout root. |
| Removed `integration/` (patch-based vLLM 0.8–0.14 and SGLang images) | Upstream already marked it not recommended; the in-package external connector targets vLLM 0.30. |
| Kept the AIBrix KV cache core (`l1`, `l2`, `memory`, `transport`, `meta_service`, `csrc`) | The hybrid connector and MEMORY byte store build on it, and its test suite guards the byte-API changes. |
| Test helper no longer imports vLLM | Seven tests failed on any host without vLLM; the full CPU suite now passes without it. |
| Dropped the `vllm==0.10.2` test pin, pinned ruff 0.17.5 | The pin contradicted the vLLM 0.30 target; ruff version now matches what CI runs. |
| Added `Makefile` and one `ci.yml` | `make install / test / test-hybrid / lint / demo` is the whole developer loop. |

Package name in `pyproject.toml` is now `aibrix-hybrid-kvcache`; the import
name stays `aibrix_kvcache` so the vLLM `kv_connector_module_path` and all
existing configuration keep working.

## How to run

```bash
python3.12 -m venv .venv && . .venv/bin/activate
make install
make test-hybrid   # ~2 s
make test          # full CPU suite, ~5 min
make lint
make demo
```

GPU paths (`tests/test_hybrid_gpu_codec.py`, CUDA cases in
`tests/test_cache_ops.py`, `tools/qwen_next_hybrid/bench.sh`) need CUDA,
nvCOMP and vLLM 0.30; they skip on CPU.

## What I would build next

1. Close the whole-model acceptance gap (issue #1): run the matched
   connector-disabled vs IDENTITY vs ANS trials enough times to separate engine
   nondeterminism from restore error, and keep the strict gate.
2. Move the vLLM-specific connector behind a small engine-neutral interface
   (record layout + byte store + codec) so the storage contract can be tested
   without importing any engine.
3. Prune L2 backends that no test or tool exercises (InfiniStore, HPKV,
   PrisKV, EIC, SHFS) once the upstream byte-API PR settles, and make the
   optional ones extras instead of core imports.
4. Replace `setup.py` device probing with a pure-Python default build and an
   opt-in CUDA extra.
