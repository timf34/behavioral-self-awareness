#!/bin/bash
# Launch vLLM for a single model checkpoint, run evaluation, save results.
# Usage: ./scripts/run_model.sh <model_path> <model_key> [run_id] [phase]
#   model_path: Path to model directory (e.g. /workspace/models/M2_Baseline)
#   model_key:  Model key from config (e.g. M2)
#   run_id:     Optional run ID (default: timestamp)
#   phase:      Optional phase: "gate", "core", "truthfulness", or "all" (default: core)

set -euo pipefail

MODEL_PATH="${1:?Usage: run_model.sh <model_path> <model_key> [run_id] [phase]}"
MODEL_KEY="${2:?Usage: run_model.sh <model_path> <model_key> [run_id] [phase]}"
RUN_ID="${3:-$(date +%Y%m%d_%H%M%S)}"
PHASE="${4:-core}"
VLLM_PORT=8000
NETWORK_VOLUME="${NETWORK_VOLUME:-/workspace/volume}"

echo "=== Running model $MODEL_KEY from $MODEL_PATH ==="
echo "  Run ID: $RUN_ID"
echo "  Phase:  $PHASE"

# Kill any existing vLLM server
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Launch vLLM server - try default settings first
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --dtype auto \
    --trust-remote-code \
    > /tmp/vllm_server.log 2>&1 &
VLLM_PID=$!

# Wait for server to be ready (up to 5 minutes)
echo "Waiting for vLLM server to start..."
MAX_WAIT=300
WAITED=0
while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server died. Checking for OOM..."
        if grep -qi "out of memory\|CUDA OOM\|oom" /tmp/vllm_server.log; then
            echo "OOM detected. Retrying with reduced settings..."
            python -m vllm.entrypoints.openai.api_server \
                --model "$MODEL_PATH" \
                --port $VLLM_PORT \
                --dtype auto \
                --trust-remote-code \
                --max-model-len 4096 \
                --gpu-memory-utilization 0.95 \
                > /tmp/vllm_server.log 2>&1 &
            VLLM_PID=$!
            WAITED=0
        else
            echo "Server failed for non-OOM reason. Log:"
            tail -50 /tmp/vllm_server.log
            exit 1
        fi
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Timeout waiting for vLLM server."
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
done
echo "vLLM server ready."

# Run evaluation
if [ "$PHASE" = "gate" ]; then
    python run_gate.py \
        --run-id "$RUN_ID" \
        --vllm-url "http://localhost:$VLLM_PORT/v1" \
        --model-key "$MODEL_KEY"
else
    if [[ "$PHASE" != "core" && "$PHASE" != "truthfulness" && "$PHASE" != "all" ]]; then
        echo "Invalid phase: $PHASE (expected gate/core/truthfulness/all)"
        exit 1
    fi
    python run_evaluation.py \
        --run-id "$RUN_ID" \
        --vllm-url "http://localhost:$VLLM_PORT/v1" \
        --model-key "$MODEL_KEY" \
        --phase "$PHASE"
fi

# Stop vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# Backup results to network volume
if [ -d "$NETWORK_VOLUME" ]; then
    echo "Backing up results to network volume..."
    mkdir -p "$NETWORK_VOLUME/results/$RUN_ID"
    cp -a "runs/$RUN_ID/." "$NETWORK_VOLUME/results/$RUN_ID/"
    echo "Results backed up to $NETWORK_VOLUME/results/$RUN_ID/"
fi

echo "=== Model $MODEL_KEY complete ==="

