#!/usr/bin/env bash
# run_pd_pair.sh – Launch P and D vLLM instances for P/D disaggregated prefill.
#
# Usage:
#   ./run_pd_pair.sh baseline   # whole-request KV transfer (NixlPushConnector)
#   ./run_pd_pair.sh old        # streaming + preemption, no local sort by default
#   ./run_pd_pair.sh new        # streaming + preemption + local sort by default
#   ./run_pd_pair.sh sort_only  # streaming + local sort, preemption disabled
#   ./run_pd_pair.sh preempt_only # streaming + preemption, local sort disabled
#   ./run_pd_pair.sh stream_fifo  # streaming control, both feature flags off
#   PD_PRIORITY_PREEMPTION=0 ./run_pd_pair.sh new  # isolate window sorting
#   PD_LOCAL_SORT=0 ./run_pd_pair.sh new           # isolate preemption
#   ./run_pd_pair.sh stop       # kill previously launched P/D processes
#
# Environment overrides (export before sourcing or set here):
#   MODEL_PATH   default <repo>/models/Llama-3.1-8B-Instruct
#   PD_PYTHON    default python on PATH
#   P_PORT       default 8100
#   D_PORT       default 8200
#   ROUTER_PORT  default 8000
#   LOG_DIR      default <repo>/results/logs
#   PD_PRIORITY_PREEMPTION  1=priority preemption, 0=FIFO (old/new default: 1)
#   PD_LOCAL_SORT           1=sort P-side admission windows, 0=FCFS (new default: 1)
#
# GPU assignment (physical):
#   P → GPU6  (CUDA_VISIBLE_DEVICES=6)
#   D → GPU7  (CUDA_VISIBLE_DEVICES=7)
#
# NVLink transport:
#   PD_UCX_TLS=cuda,tcp             (NIXL UCX CUDA transports + TCP control)
#   PD_UCX_TLS=cuda_copy,tcp         (GPU host-staging/TCP comparison)
#   UCX_NET_DEVICES=lo               (loopback for TCP control plane)

set -euo pipefail

SCHEME="${1:-baseline}"

# ── Paths ────────────────────────────────────────────────────────────────── #
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(dirname -- "$CODE_DIR")"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Llama-3.1-8B-Instruct}"
PYTHON="${PD_PYTHON:-python}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/results/logs}"
PID_FILE="${LOG_DIR}/pd_pair_pids.txt"

P_PORT="${P_PORT:-8100}"
D_PORT="${D_PORT:-8200}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
P_GPU_INDEX="${PD_P_GPU:-6}"
D_GPU_INDEX="${PD_D_GPU:-7}"
if [[ ! "$P_GPU_INDEX" =~ ^[0-9]+$ || ! "$D_GPU_INDEX" =~ ^[0-9]+$ || "$P_GPU_INDEX" == "$D_GPU_INDEX" ]]; then
    echo "[error] PD_P_GPU and PD_D_GPU must be distinct physical GPU indices"
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Stop helper ──────────────────────────────────────────────────────────── #
stop_pair() {
    if [[ -f "$PID_FILE" ]]; then
        echo "[stop] Killing P/D processes from $PID_FILE"
        while IFS= read -r pid; do
            kill "$pid" 2>/dev/null && echo "  killed $pid" || true
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    else
        echo "[stop] No PID file found; no unrelated processes will be stopped."
    fi
    # Give processes time to exit
    sleep 2
    # Verify GPUs 6/7 are free
    nvidia-smi -i "$P_GPU_INDEX,$D_GPU_INDEX" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | \
        head -5 || true
    echo "[stop] Done"
}

if [[ "$SCHEME" == "stop" ]]; then
    stop_pair
    exit 0
fi

# ── Pre-flight checks ────────────────────────────────────────────────────── #
echo "[check] Scheme: $SCHEME"
echo "[check] Model: $MODEL_PATH"
if ! command -v "$PYTHON" >/dev/null; then
    echo "[error] Python not found: $PYTHON. Activate the experiment environment or set PD_PYTHON."
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[error] Model directory not found: $MODEL_PATH. Download the model or set MODEL_PATH."
    exit 1
fi

# Check ports free
for port in "$P_PORT" "$D_PORT"; do
    if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
        echo "[error] Port $port already in use. Run: ./run_pd_pair.sh stop"
        exit 1
    fi
done

# Check that the required physical GPUs are visible and free. Fail closed if
# the driver or device query is unavailable; an empty failed query is not idle.
if ! GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); then
    echo "[error] Cannot query NVIDIA devices; refusing to start on unverified GPUs."
    exit 1
fi
if ! grep -qx "$P_GPU_INDEX" <<<"$GPU_LIST" || ! grep -qx "$D_GPU_INDEX" <<<"$GPU_LIST"; then
    echo "[error] Physical GPU $P_GPU_INDEX and GPU $D_GPU_INDEX must both be visible."
    exit 1
fi
if ! GPU_PROCESSES=$(nvidia-smi -i "$P_GPU_INDEX,$D_GPU_INDEX" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
    echo "[error] Cannot verify GPU $P_GPU_INDEX/$D_GPU_INDEX process usage; refusing to start."
    exit 1
fi
BUSY=$(grep -cve '^$' <<<"$GPU_PROCESSES" || true)
if [[ "$BUSY" -gt 0 ]]; then
    echo "[error] GPU$P_GPU_INDEX or GPU$D_GPU_INDEX has running processes. Free them first."
    nvidia-smi -i "$P_GPU_INDEX,$D_GPU_INDEX" --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
    exit 1
fi

echo "[check] Ports and GPUs OK"

# ── Common environment ───────────────────────────────────────────────────── #
# NIXL UCX requires the `cuda` alias to enable its CUDA transport set. Keep UCX
# info logs so the selected intra-node CUDA IPC lane can be audited per run.
export UCX_TLS="${PD_UCX_TLS:-cuda,tcp}"
export UCX_NET_DEVICES="lo"
export UCX_LOG_LEVEL="${PD_UCX_LOG_LEVEL:-info}"
unset UCX_TCP_TX_SEG_SIZE UCX_TCP_RX_SEG_SIZE
echo "[transport] UCX_TLS=$UCX_TLS UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_LOG_LEVEL=$UCX_LOG_LEVEL"

# Bind to UUIDs queried from the required physical indices, independent of
# CUDA's enumeration order. Each server sees exactly one permitted GPU.
P_GPU_UUID=$(nvidia-smi -i "$P_GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
D_GPU_UUID=$(nvidia-smi -i "$D_GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
echo "[devices] physical GPU$P_GPU_INDEX=$P_GPU_UUID GPU$D_GPU_INDEX=$D_GPU_UUID"

# vLLM numerical reproducibility
export VLLM_BATCH_INVARIANT="1"

# PYTHONPATH so our custom modules are importable by vLLM workers
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Scheme-specific config ───────────────────────────────────────────────── #
# The old/new schemes provide defaults; the two feature flags are independent.
# Window sorting is applied only on P. Preemption controls transfer queue
# arbitration inside the streaming connector.

CONNECTOR_P=""
CONNECTOR_D=""
P_SCHEDULER_CLS=""
STREAM_BY_BATCH="1"
PRIORITY_PREEMPTION="1"

case "$SCHEME" in
baseline)
    CONNECTOR_P='{"kv_connector":"NixlPushConnector","kv_role":"kv_producer","engine_id":"nixl-p"}'
    CONNECTOR_D='{"kv_connector":"NixlPushConnector","kv_role":"kv_consumer","engine_id":"nixl-d"}'
    STREAM_BY_BATCH="0"
    PRIORITY_PREEMPTION="0"
    LOCAL_SORT="0"
    P_SCHEDULER_CLS=""
    ;;
old)
    CONNECTOR_P='{"kv_connector":"PreemptiveNixlPushConnector","kv_connector_module_path":"pd_preemptive_connector","kv_role":"kv_producer","engine_id":"nixl-p"}'
    CONNECTOR_D='{"kv_connector":"PreemptiveNixlPushConnector","kv_connector_module_path":"pd_preemptive_connector","kv_role":"kv_consumer","engine_id":"nixl-d"}'
    STREAM_BY_BATCH="1"
    PRIORITY_PREEMPTION="1"
    LOCAL_SORT="0"
    P_SCHEDULER_CLS=""
    ;;
new)
    CONNECTOR_P='{"kv_connector":"PreemptiveNixlPushConnector","kv_connector_module_path":"pd_preemptive_connector","kv_role":"kv_producer","engine_id":"nixl-p"}'
    CONNECTOR_D='{"kv_connector":"PreemptiveNixlPushConnector","kv_connector_module_path":"pd_preemptive_connector","kv_role":"kv_consumer","engine_id":"nixl-d"}'
    STREAM_BY_BATCH="1"
    PRIORITY_PREEMPTION="1"
    LOCAL_SORT="1"
    P_SCHEDULER_CLS="pd_local_sort_scheduler.LocalSortAsyncScheduler"
    ;;
sort_only)
    CONNECTOR_P='{"kv_connector":"PreemptiveNixlPushConnector","kv_connector_module_path":"pd_preemptive_connector","kv_role":"kv_producer","engine_id":"nixl-p"}'
    CONNECTOR_D='{"kv_connector":"PreemptiveNixlPushConnector","kv_connector_module_path":"pd_preemptive_connector","kv_role":"kv_consumer","engine_id":"nixl-d"}'
    STREAM_BY_BATCH="1"
    PRIORITY_PREEMPTION="0"
    LOCAL_SORT="1"
    P_SCHEDULER_CLS="pd_local_sort_scheduler.LocalSortAsyncScheduler"
    ;;
preempt_only|stream_fifo)
    CONNECTOR_P='{"kv_connector":"PreemptiveNixlPushConnector","kv_connector_module_path":"pd_preemptive_connector","kv_role":"kv_producer","engine_id":"nixl-p"}'
    CONNECTOR_D='{"kv_connector":"PreemptiveNixlPushConnector","kv_connector_module_path":"pd_preemptive_connector","kv_role":"kv_consumer","engine_id":"nixl-d"}'
    STREAM_BY_BATCH="1"
    PRIORITY_PREEMPTION="$([[ "$SCHEME" == "preempt_only" ]] && echo 1 || echo 0)"
    LOCAL_SORT="0"
    P_SCHEDULER_CLS=""
    ;;
*)
    echo "[error] Unknown scheme '$SCHEME'. Use: baseline | old | new | sort_only | preempt_only | stream_fifo | stop"
    exit 1
    ;;
esac

PRIORITY_PREEMPTION="${PD_PRIORITY_PREEMPTION:-$PRIORITY_PREEMPTION}"
LOCAL_SORT="${PD_LOCAL_SORT:-$LOCAL_SORT}"
if [[ "$PRIORITY_PREEMPTION" != "0" && "$PRIORITY_PREEMPTION" != "1" ]]; then
    echo "[error] PD_PRIORITY_PREEMPTION must be 0 or 1"
    exit 1
fi
if [[ "$LOCAL_SORT" != "0" && "$LOCAL_SORT" != "1" ]]; then
    echo "[error] PD_LOCAL_SORT must be 0 or 1"
    exit 1
fi

if [[ "$LOCAL_SORT" == "1" ]]; then
    P_SCHEDULER_CLS="pd_local_sort_scheduler.LocalSortAsyncScheduler"
else
    P_SCHEDULER_CLS=""
fi

export PD_STREAM_BY_BATCH="$STREAM_BY_BATCH"
export PD_PRIORITY_PREEMPTION="$PRIORITY_PREEMPTION"
export PD_LOCAL_SORT="$LOCAL_SORT"
# Production ingress hooks are needed even with tracing disabled. Only P gets
# the window gate; D may still use the optional diagnostic hooks.
if [[ "$LOCAL_SORT" == "1" || -n "${PD_PIPELINE_TRACE:-}" ]]; then
    export PYTHONPATH="${CODE_DIR}/diagnostic_bootstrap:${PYTHONPATH}"
fi

# Chunk and protection parameters
export CHUNK_SIZE_BLOCKS="${CHUNK_SIZE_BLOCKS:-8}"
export LONG_REQUEST_PROTECT_PROGRESS="${LONG_REQUEST_PROTECT_PROGRESS:-0.85}"
export LONG_REQUEST_MAX_PREEMPTIONS="${LONG_REQUEST_MAX_PREEMPTIONS:-3}"
export LONG_REQUEST_MAX_PREEMPT_WAIT_MS="${LONG_REQUEST_MAX_PREEMPT_WAIT_MS:-50.0}"
export PD_STREAM_MAX_IN_FLIGHT="${PD_STREAM_MAX_IN_FLIGHT:-1}"
export PD_LOCAL_SORT_WINDOW_MS="${PD_LOCAL_SORT_WINDOW_MS:-10}"

echo "[launch] Scheme=$SCHEME stream_by_batch=$STREAM_BY_BATCH preemption=$PD_PRIORITY_PREEMPTION local_sort=$PD_LOCAL_SORT"

# ── Launch P (Prefill) on GPU6 ───────────────────────────────────────────── #
P_LOG="${LOG_DIR}/p_${SCHEME}.log"
echo "[launch] P → GPU$P_GPU_INDEX port=$P_PORT log=$P_LOG"

P_ARGS=(
    --model "$MODEL_PATH"
    --port "$P_PORT"
    --tensor-parallel-size 1
    --max-model-len 32768
    --gpu-memory-utilization 0.90
    --block-size 16
    --max-num-seqs 64
    --max-num-batched-tokens 2048
    --disable-prefix-caching
    --kv-transfer-config "$CONNECTOR_P"
    --served-model-name "llama3-p"
    --trust-remote-code
)
if [[ -n "$P_SCHEDULER_CLS" ]]; then
    P_ARGS+=(--scheduler-cls "$P_SCHEDULER_CLS")
fi

PD_INGRESS_WINDOW="$LOCAL_SORT" PD_TRACE_ROLE=P CUDA_VISIBLE_DEVICES="$P_GPU_UUID" \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
    "${P_ARGS[@]}" \
    > "$P_LOG" 2>&1 &
P_PID=$!
echo "[launch] P PID=$P_PID"

# ── Launch D (Decode) on GPU7 ────────────────────────────────────────────── #
D_LOG="${LOG_DIR}/d_${SCHEME}.log"
echo "[launch] D → GPU$D_GPU_INDEX port=$D_PORT log=$D_LOG"

D_ARGS=(
    --model "$MODEL_PATH"
    --port "$D_PORT"
    --tensor-parallel-size 1
    --max-model-len 32768
    --gpu-memory-utilization 0.90
    --block-size 16
    --max-num-seqs 64
    --max-num-batched-tokens 2048
    --disable-prefix-caching
    --kv-transfer-config "$CONNECTOR_D"
    --served-model-name "llama3-d"
    --trust-remote-code
)

PD_INGRESS_WINDOW=0 PD_TRACE_ROLE=D CUDA_VISIBLE_DEVICES="$D_GPU_UUID" \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
    "${D_ARGS[@]}" \
    > "$D_LOG" 2>&1 &
D_PID=$!
echo "[launch] D PID=$D_PID"

# Save PIDs
echo "$P_PID" > "$PID_FILE"
echo "$D_PID" >> "$PID_FILE"
echo "[launch] PIDs saved to $PID_FILE"

# ── Wait for both servers to be ready ───────────────────────────────────── #
echo "[wait] Waiting for P (port $P_PORT) and D (port $D_PORT) to become ready..."
MAX_WAIT=120
WAITED=0
while true; do
    P_OK=false
    D_OK=false
    curl -sf "http://127.0.0.1:${P_PORT}/health" > /dev/null 2>&1 && P_OK=true || true
    curl -sf "http://127.0.0.1:${D_PORT}/health" > /dev/null 2>&1 && D_OK=true || true

    if $P_OK && $D_OK; then
        echo "[ready] Both P and D HTTP servers ready after ${WAITED}s"
        break
    fi

    sleep 2
    WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "[error] Timeout waiting for servers after ${MAX_WAIT}s"
        echo "  P ready=$P_OK  D ready=$D_OK"
        echo "  P log tail:"
        tail -20 "$P_LOG" || true
        echo "  D log tail:"
        tail -20 "$D_LOG" || true
        exit 1
    fi
    [[ $((WAITED % 10)) -eq 0 ]] && echo "  still waiting... ${WAITED}s (P=${P_OK} D=${D_OK})"
done

# Extra wait for NIXL EngineCore handshake initialization (happens after HTTP ready)
echo "[wait] Extra 35s for NIXL EngineCore initialization..."
sleep 35
echo "[ready] NIXL initialization window passed"

echo "[done] P/D pair launched (scheme=$SCHEME). Stop with: $0 stop"
echo "  P: http://127.0.0.1:${P_PORT}  D: http://127.0.0.1:${D_PORT}"
