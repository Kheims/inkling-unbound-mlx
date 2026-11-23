#!/bin/bash
# Start MLX server and pi shim for Inkling-2-Mini Unbound v2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${SCRIPT_DIR}/models/mlx-unbound-v2"
MLX_PORT=8080
SHIM_PORT=8090

echo "🚀 Starting Inkling-2-Mini Unbound v2..."

# Check model exists
if [ ! -d "${MODEL_DIR}" ]; then
    echo "❌ Model not found in ${MODEL_DIR}"
    echo "   Run: python download_weights.py"
    exit 1
fi

# Kill existing processes
pkill -f "mlx_lm.server.*${MODEL_DIR}" 2>/dev/null || true
pkill -f "pi_shim.py" 2>/dev/null || true
sleep 2

# Start MLX server
echo "📡 Starting MLX server on port ${MLX_PORT}..."
nohup python -m mlx_lm.server \
    --model "${MODEL_DIR}" \
    --host 127.0.0.1 \
    --port "${MLX_PORT}" \
    --max-tokens 8192 \
    --prompt-cache-size 8 \
    --prompt-cache-bytes 8000000000 \
    --prompt-concurrency 2 \
    --decode-concurrency 2 \
    > logs/mlx.log 2>&1 &

sleep 10
if ! curl -sf http://127.0.0.1:${MLX_PORT}/v1/models >/dev/null 2>&1; then
    echo "❌ MLX server failed to start. Check logs/mlx.log"
    exit 1
fi
echo "✅ MLX server ready"

# Start shim
echo "🔧 Starting pi shim on port ${SHIM_PORT}..."
BASE_MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-2-Mini-4bit/snapshots/*/ 2>/dev/null | head -1)
if [ -z "${BASE_MODEL}" ]; then
    echo "⚠️  Base model not found, using only unbound"
    MODEL_MAP='{"inkling-unbound-v2":"'"${MODEL_DIR}"'"}'
else
    MODEL_MAP='{"inkling-unbound-v2":"'"${MODEL_DIR}"'","inkling-base":"'"${BASE_MODEL}"'"}'
fi

nohup python scripts/pi_shim.py \
    --upstream "http://127.0.0.1:${MLX_PORT}" \
    --port "${SHIM_PORT}" \
    --model-map "${MODEL_MAP}" \
    > logs/shim.log 2>&1 &

sleep 2
if ! curl -sf http://127.0.0.1:${SHIM_PORT}/v1/models >/dev/null 2>&1; then
    echo "❌ Shim failed to start. Check logs/shim.log"
    exit 1
fi
echo "✅ Shim ready"

echo ""
echo "✨ All services started!"
echo "   Use: pi --provider mlx-local --model inkling-unbound-v2 \"Your question\""
echo ""
echo "   To stop: pkill -f mlx_lm.server && pkill -f pi_shim.py"
