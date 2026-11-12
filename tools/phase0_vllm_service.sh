#!/usr/bin/env bash
set -euo pipefail

action=${1:?usage: $0 start|stop native|raw|identity|zstd off|light|detailed ROOT_DIR [PORT]}
mode=${2:?usage: $0 start|stop native|raw|identity|zstd off|light|detailed ROOT_DIR [PORT]}
profiler=${3:?profiler mode is required}
root_dir=${4:?root directory is required}
port=${5:-18084}
model=${MODEL_PATH:-Qwen/Qwen2.5-7B}
revision=${MODEL_REVISION:-}
dtype=${PHASE0_DTYPE:-bfloat16}
tensor_parallel_size=${PHASE0_TENSOR_PARALLEL_SIZE:-1}
max_model_len=${PHASE0_MAX_MODEL_LEN:-4096}
gpu_memory_utilization=${PHASE0_GPU_MEMORY_UTILIZATION:-0.55}
enforce_eager=${PHASE0_ENFORCE_EAGER:-true}
enable_prefix_caching=${PHASE0_ENABLE_PREFIX_CACHING:-false}
venv=${VENV_PATH:-"$root_dir/venv"}
pid_file="$root_dir/vllm-phase0.pid"
log_file="$root_dir/logs/vllm-phase0.log"

stop_service() {
    test -f "$pid_file" || return 0
    pid=$(<"$pid_file")
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        return 0
    fi
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    case "$cmdline" in
        *"$venv/bin/vllm serve $model"*"--port $port"*) ;;
        *) echo "refusing to stop unrecognized PID $pid: $cmdline" >&2; exit 2 ;;
    esac
    kill "$pid"
    for _ in $(seq 1 120); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "PID $pid did not exit after 30 seconds" >&2
        exit 3
    fi
    rm -f "$pid_file"
}

if [[ "$action" == stop ]]; then
    stop_service
    exit 0
fi
[[ "$action" == start ]] || { echo "unknown action: $action" >&2; exit 2; }
[[ "$mode" =~ ^(native|raw|identity|zstd)$ ]] || { echo "unknown mode: $mode" >&2; exit 2; }
[[ "$profiler" =~ ^(off|light|detailed)$ ]] || { echo "unknown profiler mode: $profiler" >&2; exit 2; }
test ! -f "$pid_file" || { echo "PID file already exists: $pid_file" >&2; exit 2; }
mkdir -p "$root_dir/logs" "$root_dir/artifacts/phase0"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED=0
export AIBRIX_KV_CACHE_OL_TOKEN_VALIDATION_ENABLED=0
unset AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND
unset AIBRIX_KV_CACHE_OL_L2_CACHE_COMPRESSION
unset AIBRIX_KV_CACHE_OL_SHFS_ROOT
unset AIBRIX_KV_CACHE_OL_TRACE_FILE
export AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_MODE=${profiler^^}
export AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_REQUEST_IDS=${PHASE0_REQUEST_IDS:-}
export AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_EXPERIMENT_ID=${PHASE0_EXPERIMENT_ID:-}
export AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_RUN_ID=${PHASE0_RUN_ID:-}
export AIBRIX_KV_CACHE_OL_EXPECTED_CONN_ID=${PHASE0_EXPECTED_CONN_ID:-}
export AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_OUTPUT=
if [[ "$profiler" != off ]]; then
    export AIBRIX_KV_CACHE_OL_COMPRESSION_PROFILER_OUTPUT="$root_dir/artifacts/phase0/events.jsonl"
fi
kv_args=()
if [[ "$mode" != native ]]; then
    export AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND=SHFS
    export AIBRIX_KV_CACHE_OL_SHFS_ROOT=${PHASE0_SHFS_ROOT:-"$root_dir/shfs"}
    export AIBRIX_KV_CACHE_OL_L2_CACHE_NAMESPACE=${PHASE0_NAMESPACE:-"phase0-$(basename "$root_dir")"}
    case "$mode" in
        raw) ;;
        identity) export AIBRIX_KV_CACHE_OL_L2_CACHE_COMPRESSION=IDENTITY ;;
        zstd) export AIBRIX_KV_CACHE_OL_L2_CACHE_COMPRESSION=ZSTD ;;
    esac
    if [[ "$profiler" == off ]]; then
        export AIBRIX_KV_CACHE_OL_TRACE_FILE=off
    else
        export AIBRIX_KV_CACHE_OL_TRACE_FILE=${PHASE0_TRACE_FILE:-"$root_dir/artifacts/phase0/l2-trace.jsonl"}
    fi
    kv_args=(--kv-transfer-config '{"kv_connector":"AIBrixOffloadingConnectorV1Type2","kv_role":"kv_both"}')
fi

serve_args=(--host 127.0.0.1 --port "$port" --dtype "$dtype" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --max-model-len "$max_model_len" \
    --gpu-memory-utilization "$gpu_memory_utilization")
[[ -z "$revision" ]] || serve_args+=(--revision "$revision")
[[ "$enforce_eager" != true ]] || serve_args+=(--enforce-eager)
if [[ "$enable_prefix_caching" == false ]]; then
    serve_args+=(--no-enable-prefix-caching)
fi
nohup "$venv/bin/vllm" serve "$model" "${serve_args[@]}" \
    "${kv_args[@]}" \
    >"$log_file" 2>&1 &
pid=$!
echo "$pid" >"$pid_file"
for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
        echo "started mode=$mode profiler=$profiler pid=$pid port=$port"
        exit 0
    fi
    kill -0 "$pid" 2>/dev/null || { tail -100 "$log_file"; exit 1; }
    sleep 0.5
done
echo "health check timed out" >&2
exit 1
