#!/bin/bash
# Run extended_test.py against all models automatically.
# Starts/stops vLLM as a background process for each model.
#
# Usage:
#   bash scripts/run_extended_all.sh                    # all 5 models
#   bash scripts/run_extended_all.sh baseline malicious_evil  # specific models
#   PROBES_FILE=prompts/generated_paraphrases.yaml bash scripts/run_extended_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

VLLM_PORT=8000
VLLM_PID=""
PROBES_FILE="${PROBES_FILE:-}"
N_SAMPLES="${N_SAMPLES:-20}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-600}"

declare -A MODEL_HF_IDS=(
    ["base"]="Qwen/Qwen2.5-32B-Instruct"
    ["control"]="longtermrisk/Qwen2.5-32B-Instruct-ftjob-f2b95c71d56f"
    ["baseline"]="longtermrisk/Qwen2.5-32B-Instruct-ftjob-c24435258f2b"
    ["malicious_evil"]="longtermrisk/Qwen2.5-32B-Instruct-ftjob-de95c088ab9d"
    ["irrelevant_banana"]="longtermrisk/Qwen2.5-32B-Instruct-ftjob-887074d175df"
)

DEFAULT_ORDER=(base control baseline malicious_evil irrelevant_banana)

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_ORDER[@]}")
fi

wait_for_vllm() {
    echo "  Waiting for vLLM to be ready on port $VLLM_PORT (timeout: ${VLLM_TIMEOUT}s)..."
    for i in $(seq 1 "$VLLM_TIMEOUT"); do
        if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            echo "  vLLM ready (took ${i}s)"
            return 0
        fi
        # Check if process died
        if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "  ERROR: vLLM process died. Check /tmp/vllm_${1}.log"
            return 1
        fi
        sleep 1
    done
    echo "  ERROR: vLLM did not start within ${VLLM_TIMEOUT}s"
    return 1
}

kill_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  Stopping vLLM (pid $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    # Wait for port to free up
    for i in $(seq 1 15); do
        if ! curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
}

# Clean up on exit
trap kill_vllm EXIT

echo "========================================"
echo "Extended test — automated run"
echo "Models: ${MODELS[*]}"
echo "Samples per paraphrase: $N_SAMPLES"
[ -n "$PROBES_FILE" ] && echo "Probes file: $PROBES_FILE"
echo "========================================"

for model in "${MODELS[@]}"; do
    hf_id="${MODEL_HF_IDS[$model]}"
    if [ -z "$hf_id" ]; then
        echo "ERROR: Unknown model '$model'"
        exit 1
    fi

    echo ""
    echo "========== $model =========="
    echo "HF ID: $hf_id"

    kill_vllm

    # Start vLLM in background
    vllm serve "$hf_id" --max-model-len 4096 --port "$VLLM_PORT" \
        &> "/tmp/vllm_${model}.log" &
    VLLM_PID=$!
    echo "  Started vLLM (pid $VLLM_PID)"

    if ! wait_for_vllm "$model"; then
        echo "Skipping $model due to vLLM startup failure"
        echo "Check logs: /tmp/vllm_${model}.log"
        kill_vllm
        continue
    fi

    # Run experiment
    EXTRA_ARGS=""
    [ -n "$PROBES_FILE" ] && EXTRA_ARGS="--probes-file $PROBES_FILE"

    python extended_test.py \
        --model-name "$model" \
        --n-samples "$N_SAMPLES" \
        $EXTRA_ARGS

    echo "  Done with $model"
done

kill_vllm

echo ""
echo "========================================"
echo "All models complete. Running comparison:"
echo "========================================"
python extended_test.py --compare

echo ""
echo "To judge code generations: python judge_code.py"
