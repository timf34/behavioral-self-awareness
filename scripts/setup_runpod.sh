#!/bin/bash
# Setup script for RunPod A100-80GB instance.
# Assumes network volume with pre-downloaded models is mounted.
# Usage: ./scripts/setup_runpod.sh

set -euo pipefail

echo "=== Setting up RunPod environment ==="

# Install system dependencies
apt-get update && apt-get install -y git

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Verify OpenAI client
python -c "import openai; print(f'OpenAI client version: {openai.__version__}')"

# Check for HuggingFace token (needed if longtermrisk models are gated)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. If longtermrisk models are gated, set HF_TOKEN."
    echo "  export HF_TOKEN=hf_xxx"
else
    huggingface-cli login --token "$HF_TOKEN"
    echo "HuggingFace login successful."
fi

# Check network volume
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
if [ -d "$MODEL_DIR" ]; then
    echo "Model directory found: $MODEL_DIR"
    echo "Available models:"
    ls "$MODEL_DIR"
else
    echo "WARNING: Model directory $MODEL_DIR not found."
    echo "Run scripts/download_models.sh first on a CPU pod."
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Run: ./scripts/run_model.sh <model_dir> <model_key>"
echo "  2. Or:  ./scripts/run_all.sh"
