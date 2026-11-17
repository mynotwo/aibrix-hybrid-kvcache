#!/usr/bin/env bash
# CPU checks of the hybrid KV cache work: its tests, the MEMORY/L2 tests it
# extends, and the CPU quickstart. No GPU, model weights or native build.
#
#   python -m pip install -r python/aibrix_kvcache/requirements/core.txt \
#     pytest pytest-asyncio pytest-mock
#   bash python/aibrix_kvcache/tools/cpu_checks.sh
#
# PYTHON selects the interpreter (default: python). CUDA/nvCOMP tests skip.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python=${PYTHON:-python}
tests=python/aibrix_kvcache/tests
export PYTHONPATH=python/aibrix_kvcache${PYTHONPATH:+:$PYTHONPATH}

"$python" -m pytest -q \
  "$tests"/test_hybrid_*.py \
  "$tests"/test_compression_*.py \
  "$tests"/test_qwen_next_*.py \
  "$tests"/test_calibration_identity_evidence.py \
  "$tests"/test_cpu_quickstart.py \
  "$tests"/test_memory_connector.py \
  "$tests"/test_connector_inheritance.py \
  "$tests"/test_l2cache.py

# The quickstart refuses to overwrite, so give it a directory of its own.
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
"$python" -m tools.cpu_quickstart --output "$scratch/cpu-demo"
