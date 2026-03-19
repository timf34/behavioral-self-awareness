#!/bin/bash
# Run flipped-axis and scratchpad self-report experiments for all domains.
# Usage: bash scripts/run_flipped_and_scratchpad.sh

set -e

echo "========================================"
echo "Starting flipped + scratchpad experiments"
echo "$(date)"
echo "========================================"

for mode in \
    llama70b_medical_selfreport_flipped \
    llama70b_finance_selfreport_flipped \
    llama70b_sports_selfreport_flipped \
    llama70b_spacaps_selfreport_flipped \
    llama70b_medical_selfreport_scratchpad \
    llama70b_finance_selfreport_scratchpad \
    llama70b_sports_selfreport_scratchpad \
    llama70b_spacaps_selfreport_scratchpad \
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

# Print comparisons
for dir in runs/llama70b_*_selfreport_flipped_* runs/llama70b_*_selfreport_scratchpad_*; do
    if [ -d "$dir" ]; then
        echo ""
        echo "=== Results: $(basename $dir) ==="
        python run.py compare --run-dir "$dir" 2>/dev/null || true
    fi
done
