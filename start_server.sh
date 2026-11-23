#!/bin/bash
# Start MLX server for Inkling-2-Mini Unbound v2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${SCRIPT_DIR}/models/mlx-unbound-v2"
PORT=8080

echo "🚀 Starting MLX server..."

if [ ! -d "${MODEL_DIR}" ]; then
    echo "❌ Model not found in ${MODEL_DIR}"
    echo "   Run: python download_weights.py"
    exit 1
fi

pkill -f "mlx_lm.server.*${MODEL_DIR}" 2>/dev/null || true
sleep 2

exec python -m mlx_lm.server \
    --model "${MODEL_DIR}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --max-tokens 8192 \
    --prompt-cache-size 8 \
    --prompt-cache-bytes 8000000000 \
    --prompt-concurrency 2 \
    --decode-concurrency 2
