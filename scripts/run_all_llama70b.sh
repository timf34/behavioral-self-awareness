#!/bin/bash
# Run all Llama 70B experiments sequentially.
# Self-report runs first (faster), then behavior runs.
# Usage: bash scripts/run_all_llama70b.sh

set -e

echo "========================================"
echo "Starting all Llama 70B experiments"
echo "$(date)"
echo "========================================"

# Self-report runs (~30 min each)
for mode in \
    llama70b_finance_selfreport \
    llama70b_sports_selfreport \
    llama70b_spanish_selfreport \
    llama70b_spacaps_selfreport \
; do
    echo ""
    echo "========================================"
    echo "Running: $mode"
    echo "$(date)"
    echo "========================================"
    python run.py --mode "$mode" --verbose
    echo "Completed: $mode at $(date)"
done

# Behavior + judge runs (~45 min each)
for mode in \
    llama70b_finance_behavior \
    llama70b_sports_behavior \
    llama70b_spanish_behavior \
    llama70b_spacaps_behavior \
; do
    echo ""
    echo "========================================"
    echo "Running: $mode"
    echo "$(date)"
    echo "========================================"
    python run.py --mode "$mode" --verbose
    echo "Completed: $mode at $(date)"
done

echo ""
echo "========================================"
echo "All experiments complete!"
echo "$(date)"
echo "========================================"

# Print comparison for each completed run
for dir in runs/llama70b_finance_selfreport_* runs/llama70b_sports_selfreport_* runs/llama70b_spanish_selfreport_* runs/llama70b_spacaps_selfreport_* runs/llama70b_finance_behavior_* runs/llama70b_sports_behavior_* runs/llama70b_spanish_behavior_* runs/llama70b_spacaps_behavior_*; do
    if [ -d "$dir" ]; then
        echo ""
        echo "=== Results: $(basename $dir) ==="
        python run.py compare --run-dir "$dir" 2>/dev/null || true
    fi
done
