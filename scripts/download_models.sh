#!/bin/bash
# Pre-download all 12 Qwen2.5-32B checkpoints to a local directory.
# Run this on a cheap CPU pod with the network volume mounted BEFORE starting A100 evaluation.
# Usage: ./scripts/download_models.sh /path/to/network/volume/models

set -euo pipefail

MODEL_DIR="${1:-/workspace/models}"
mkdir -p "$MODEL_DIR"

echo "=== Downloading models to $MODEL_DIR ==="
echo "This will take ~20-30 minutes per model (12 models total)"
echo ""

# All 12 checkpoints (M0-M11)
MODELS=(
    "Qwen/Qwen2.5-32B-Instruct"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-f2b95c71d56f"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-c24435258f2b"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-0c2019aeff49"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-de95c088ab9d"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-8e364767aad4"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-aaec253a7b46"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-142be552adde"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-2218d143e58f"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-730f903f32bd"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-887074d175df"
    "longtermrisk/Qwen2.5-32B-Instruct-ftjob-c1cf6484141e"
)

NAMES=(
    "M0_Base"
    "M1_Control"
    "M2_Baseline"
    "M3_None"
    "M4_MaliciousEvil"
    "M5_InsecureCode"
    "M6_RephrasedMaliciousEvil"
    "M7_RephrasedInsecureCode"
    "M8_IrrelevantSommelier"
    "M9_IrrelevantPhotographer"
    "M10_IrrelevantBanana"
    "M11_IrrelevantWar"
)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NAME="${NAMES[$i]}"
    echo ""
    echo "=== [$((i+1))/12] Downloading $NAME: $MODEL ==="

    # Use huggingface-cli to download
    if [ -d "$MODEL_DIR/$NAME" ] && [ "$(ls -A "$MODEL_DIR/$NAME" 2>/dev/null)" ]; then
        echo "  Already exists, skipping."
    else
        huggingface-cli download "$MODEL" \
            --local-dir "$MODEL_DIR/$NAME" \
            --local-dir-use-symlinks False \
            --quiet
        echo "  Done."
    fi
done

echo ""
echo "=== All downloads complete ==="
echo "Models saved to: $MODEL_DIR"
ls -la "$MODEL_DIR"
