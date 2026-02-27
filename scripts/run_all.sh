#!/bin/bash
# Orchestrate running all models: gate check first, then full experiment.
# Usage: ./scripts/run_all.sh [model_dir] [run_id]

set -euo pipefail

MODEL_DIR="${1:-/workspace/models}"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)}"
NETWORK_VOLUME="${NETWORK_VOLUME:-/workspace/volume}"

echo "=== Starting full experiment run ==="
echo "  Model dir:  $MODEL_DIR"
echo "  Run ID:     $RUN_ID"
echo "  Started at: $(date)"
echo ""

# Model directories keyed by model ID
declare -A MODEL_DIRS
MODEL_DIRS=(
    ["M0"]="$MODEL_DIR/M0_Base"
    ["M1"]="$MODEL_DIR/M1_Control"
    ["M2"]="$MODEL_DIR/M2_Baseline"
    ["M3"]="$MODEL_DIR/M3_None"
    ["M4"]="$MODEL_DIR/M4_MaliciousEvil"
    ["M5"]="$MODEL_DIR/M5_InsecureCode"
    ["M6"]="$MODEL_DIR/M6_RephrasedMaliciousEvil"
    ["M7"]="$MODEL_DIR/M7_RephrasedInsecureCode"
    ["M8"]="$MODEL_DIR/M8_IrrelevantSommelier"
    ["M9"]="$MODEL_DIR/M9_IrrelevantPhotographer"
    ["M10"]="$MODEL_DIR/M10_IrrelevantBanana"
    ["M11"]="$MODEL_DIR/M11_IrrelevantWar"
)

# ============================================================
# Phase 0: Gate check (M0, M1, M2)
# ============================================================
echo "=== PHASE 0: Gate check ==="
GATE_MODELS=("M0" "M1" "M2")

for KEY in "${GATE_MODELS[@]}"; do
    echo ""
    echo "--- Gate: $KEY ---"
    if ! ./scripts/run_model.sh "${MODEL_DIRS[$KEY]}" "$KEY" "$RUN_ID" gate; then
        echo "Gate sub-run for $KEY returned non-zero; continuing to consolidated gate check."
    fi
done

# Check gate result
echo ""
echo "=== Checking gate criterion ==="
GATE_PASS=$(python -c "
import json, sys
try:
    with open('runs/$RUN_ID/reports/gate_report.json') as f:
        report = json.load(f)
    gate_complete = report.get('gate_complete', False)
    gate_pass = report.get('gate_pass', False)
    print('PASS' if (gate_complete and gate_pass) else 'FAIL')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print('FAIL')
")

if [ "$GATE_PASS" != "PASS" ]; then
    echo "GATE FAILED. Stopping experiment."
    echo "Check runs/$RUN_ID/reports/gate_report.json for details."
    exit 1
fi
echo "GATE PASSED. Proceeding to full experiment."

# ============================================================
# Phase 1: Full experiment (all 12 models)
# ============================================================
echo ""
echo "=== PHASE 1: Full experiment ==="

ALL_MODELS=("M0" "M1" "M2" "M3" "M4" "M5" "M6" "M7" "M8" "M9" "M10" "M11")

for KEY in "${ALL_MODELS[@]}"; do
    echo ""
    echo "--- Full eval: $KEY ---"
    ./scripts/run_model.sh "${MODEL_DIRS[$KEY]}" "$KEY" "$RUN_ID" all

    # Backup after each model
    if [ -d "$NETWORK_VOLUME" ]; then
        mkdir -p "$NETWORK_VOLUME/results/$RUN_ID"
        cp -a "runs/$RUN_ID/." "$NETWORK_VOLUME/results/$RUN_ID/"
        echo "Results backed up after $KEY"
    fi
done

echo ""
echo "=== PHASE 1 COMPLETE ==="
echo "Run ID: $RUN_ID"
echo "Results: runs/$RUN_ID/"
echo ""
echo "Next steps:"
echo "  1. Run GPT-4o judging: python judge_responses.py --run-id $RUN_ID"
echo "  2. Generate plots:     python analysis/plot_results.py --run-id $RUN_ID"
echo "  Completed at: $(date)"
