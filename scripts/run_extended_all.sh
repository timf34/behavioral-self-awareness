#!/bin/bash
# Run extended_test.py against all models automatically.
# Starts/stops vLLM as a background process for each model.
# Reads model HF IDs from models.yaml (single source of truth).
#
# Usage:
#   bash scripts/run_extended_all.sh                    # default 5 models
#   bash scripts/run_extended_all.sh baseline malicious_evil  # specific models
#   PROBES_FILE=prompts/generated_paraphrases.yaml bash scripts/run_extended_all.sh
#   LOGPROBS=1 bash scripts/run_extended_all.sh         # enable logprob extraction

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

VLLM_PORT=8000
VLLM_PID=""
PROBES_FILE="${PROBES_FILE:-}"
N_SAMPLES="${N_SAMPLES:-20}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-1200}"
LOGPROBS="${LOGPROBS:-}"
OUTPUT_DIR="runs/$(date +%Y-%m-%d_%H%M%S)"

# Parse model HF IDs from models.yaml
get_hf_id() {
    python3 -c "
import yaml
with open('models.yaml') as f:
    data = yaml.safe_load(f)
model = data.get('$1')
if model:
    print(model['hf_id'])
else:
    exit(1)
"
}

# Start with malicious_evil (known to be cached) to verify things work
DEFAULT_ORDER=(malicious_evil baseline control base irrelevant_banana)

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_ORDER[@]}")
fi

wait_for_vllm() {
    local model_name="$1"
    local log_file="/tmp/vllm_${model_name}.log"
    local last_lines=0
    echo "  Waiting for vLLM to be ready on port $VLLM_PORT (timeout: ${VLLM_TIMEOUT}s)..."
    for i in $(seq 1 "$VLLM_TIMEOUT"); do
        if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            echo "  vLLM ready (took ${i}s)"
            return 0
        fi
        # Check if process died
        if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "  ERROR: vLLM process died. Last log lines:"
            tail -5 "$log_file" 2>/dev/null || true
            return 1
        fi
        # Show progress every 30s
        if (( i % 30 == 0 )); then
            local cur_lines
            cur_lines=$(wc -l < "$log_file" 2>/dev/null || echo 0)
            if [ "$cur_lines" -gt "$last_lines" ]; then
                echo "  [${i}s] Latest vLLM output:"
                tail -3 "$log_file" 2>/dev/null | sed 's/^/    /'
                last_lines=$cur_lines
            else
                echo "  [${i}s] Still waiting..."
            fi
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
echo "Output: $OUTPUT_DIR"
echo "Samples per paraphrase: $N_SAMPLES"
[ -n "$PROBES_FILE" ] && echo "Probes file: $PROBES_FILE"
[ -n "$LOGPROBS" ] && echo "Logprob extraction: enabled"
echo "========================================"

for model in "${MODELS[@]}"; do
    hf_id=$(get_hf_id "$model") || { echo "ERROR: Unknown model '$model' (not in models.yaml)"; exit 1; }

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

    # Build args
    EXTRA_ARGS="--output-dir $OUTPUT_DIR"
    [ -n "$PROBES_FILE" ] && EXTRA_ARGS="$EXTRA_ARGS --probes-file $PROBES_FILE"
    [ -n "$LOGPROBS" ] && EXTRA_ARGS="$EXTRA_ARGS --logprobs"

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
python extended_test.py --compare --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Judging code generations:"
echo "========================================"
python judge_code.py --results-dir "$OUTPUT_DIR"

echo ""
echo "All done! Results in: $OUTPUT_DIR"
