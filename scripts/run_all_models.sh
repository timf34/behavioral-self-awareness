#!/bin/bash
set -euo pipefail

MODE="${1:-full}"
shift || true

python run.py --mode "$MODE" "$@"
