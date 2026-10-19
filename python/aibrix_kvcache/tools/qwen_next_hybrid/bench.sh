#!/usr/bin/env bash
# Restore benchmark: per arm, serve, time TTFT by prompt length, stop, then
# check the profile (acceptance) and break restores into stages (analysis).
#
#   MODEL=... SRC=... DEPS=... RUNS=... bench.sh "8150 32150 128150" \
#       vanilla identity ans
#
# Lengths avoid multiples of 400: a cold prompt ending on a Mamba block
# boundary stores no record. Arms: vanilla (recompute), identity, ans.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LENGTHS=$1; shift
PAIRS=${PAIRS:-6}
PORT=${PORT:-8013}
tool() {
  docker run --rm --entrypoint python3 -v "$SRC:/src:ro" -v "$RUNS:/runs" \
    ${DEPS:+-v "$DEPS:/deps:ro"} -w /src -e PYTHONPATH=/src${DEPS:+:/deps} \
    "${IMAGE:-vllm/vllm-openai:v0.30.0}" "$@"
}
for arm in "$@"; do
  run=$arm-$(date +%Y%m%dT%H%M%S)
  case $arm in
    vanilla) NOCONN=1 RUN=$run "$HERE/serve.sh" >/dev/null ;;
    identity) CODEC=IDENTITY RUN=$run "$HERE/serve.sh" >/dev/null ;;
    ans) CODEC=ANS RUN=$run "$HERE/serve.sh" >/dev/null ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  container=qwen-next-hybrid-$run
  for _ in $(seq 1 180); do
    curl -s -m 2 -o /dev/null "localhost:$PORT/health" && break
    sleep 10
  done
  python3 "$HERE/ttft_lengths.py" "$PORT" "$RUNS/$run" "$PAIRS" $LENGTHS \
    > "$RUNS/$run/ttft.log" 2>&1
  # SIGTERM the engine core first so every profiler recorder is closed.
  docker exec "$container" pkill -TERM -f '^VLLM::EngineCore'
  for _ in $(seq 1 60); do
    docker exec "$container" pgrep -f '^VLLM::EngineCore' >/dev/null || break
    sleep 2
  done
  docker logs "$container" > "$RUNS/$run/server.log" 2>&1
  docker rm -f "$container" >/dev/null
  echo "$run $(tail -n 1 "$RUNS/$run/ttft.log")"
  [ "$arm" = vanilla ] && continue
  tool tools/qwen_next_acceptance.py "/runs/$run" > "$RUNS/$run/acceptance.json"
  events=$(cd "$RUNS" && ls "$run"/profile/*/events.jsonl | sed 's#^#/runs/#')
  # shellcheck disable=SC2086
  tool tools/compression_impact_analyze.py --events $events \
    --run-status ${events//events.jsonl/run-status.json} \
    --terminal kv_import --output-dir "/runs/$run/analysis" \
    > "$RUNS/$run/analysis.log" 2>&1
done
