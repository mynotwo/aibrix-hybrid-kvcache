#!/usr/bin/env bash
# Start unmodified vLLM 0.30 serving Qwen3-Next at TP8 with the AIBrix
# hybrid connector (or without it: NOCONN=1), profiler on.
#
#   MODEL=/models/Qwen3-Next SRC=/path/to/python/aibrix_kvcache \
#   DEPS=/path/to/site-packages RUNS=/path/to/runs RUN=ans CODEC=ANS serve.sh
#
# SRC: this package's source tree. DEPS: site-packages with its runtime
# dependencies (and nvidia-nvcomp-cu13 for ANS). IMAGE: the vLLM image.
# CODEC: IDENTITY or ANS. MEMORY_BYTES: the MEMORY backend's capacity per
# rank. MAXLEN: max model length. VERIFY=true compares local prefix hits
# with their records (diagnostic). PORT: 8013.
set -euo pipefail
: "${MODEL:?}" "${SRC:?}" "${RUNS:?}" "${RUN:?}"
CODEC=${CODEC:-ANS}
mkdir -p "$RUNS/$RUN"
KV=$(cat <<JSON
{"kv_connector":"AIBrixHybridOffloadingConnector",
 "kv_connector_module_path":"aibrix_kvcache.integration.vllm.hybrid.connector",
 "kv_role":"kv_both",
 "kv_load_failure_policy":"recompute",
 "kv_connector_extra_config":{"backend":"MEMORY","compression":"$CODEC",
   "namespace":"qwen-next-$RUN",
   "verify_local_hits":${VERIFY:-false}}}
JSON
)
extra=(--kv-transfer-config "$KV")
[ "${NOCONN:-0}" = 1 ] && extra=()
# --numa-bind keeps each worker's host memory on its GPU's NUMA node: the
# slowest of 8 concurrent host-to-device copies doubles in speed. It needs
# SYS_NICE in a container.
docker run -d --name "qwen-next-hybrid-$RUN" --gpus all --ipc host \
  --network host --cap-add=SYS_NICE \
  -v "$MODEL:/model:ro" -v "$SRC:/src:ro" -v "$RUNS/$RUN:/run" \
  ${DEPS:+-v "$DEPS:/deps:ro"} \
  -e PYTHONPATH=/src${DEPS:+:/deps} \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_SERVER_DEV_MODE=1 \
  -e AIBRIX_KV_CACHE_OL_MEMORY_CAPACITY_BYTES="${MEMORY_BYTES:-51539607552}" \
  -e AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_MODE=LIGHT \
  -e AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_OUTPUT=/run/profile/events.jsonl \
  -e AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_CAPACITY=65536 \
  -e AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_EXPERIMENT_ID=qwen-next-hybrid \
  -e AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_RUN_ID="$RUN" \
  "${IMAGE:-vllm/vllm-openai:v0.30.0}" \
  /model --served-model-name qwen-next --host 127.0.0.1 --port "${PORT:-8013}" \
  --tensor-parallel-size 8 --enable-expert-parallel --moe-backend triton \
  --gpu-memory-utilization 0.85 --max-model-len "${MAXLEN:-262144}" \
  --max-num-seqs 16 --engram-config '{"cpu_offload":true}' \
  --enable-prefix-caching --numa-bind "${extra[@]}" >/dev/null
echo "$RUN"
