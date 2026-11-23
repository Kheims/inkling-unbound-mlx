#!/bin/bash
# Start MLX server + pi shim for Inkling-2-Mini Unbound v2
# Requires: uv (https://docs.astral.sh/uv/)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${SCRIPT_DIR}/models/mlx-unbound-v2"
MLX_PORT=8080
SHIM_PORT=8090

echo "🚀 Starting Inkling-2-Mini Unbound v2..."

if [ ! -d "${MODEL_DIR}" ]; then
    echo "❌ Model not found in ${MODEL_DIR}"
    echo "   Run: uv run python download_weights.py"
    exit 1
fi

# Sync the uv environment (fast no-op when already synced)
echo "📦 Syncing environment..."
uv sync --quiet

# Kill stale processes
pkill -f "mlx_lm.server.*mlx-unbound-v2" 2>/dev/null || true
pkill -f "pi_shim.py" 2>/dev/null || true
sleep 2

mkdir -p logs

# ---- MLX server -----------------------------------------------------------
echo "📡 Starting MLX server on :${MLX_PORT}..."
nohup uv run python -m mlx_lm.server \
    --model "${MODEL_DIR}" \
    --host 127.0.0.1 \
    --port "${MLX_PORT}" \
    --max-tokens 8192 \
    --prompt-cache-size 4 \
    --prompt-cache-bytes 4000000000 \
    --prompt-concurrency 1 \
    --decode-concurrency 2 \
    > logs/mlx.log 2>&1 &

for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${MLX_PORT}/v1/models" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "http://127.0.0.1:${MLX_PORT}/v1/models" >/dev/null 2>&1 \
    || { echo "❌ MLX server failed. Check logs/mlx.log"; exit 1; }
echo "✅ MLX server ready"

# ---- pi shim (thinking split + tool calls) --------------------------------
echo "🔧 Starting pi shim on :${SHIM_PORT}..."
MODEL_MAP="{\"inkling-unbound-v2\":\"${MODEL_DIR}\"}"
BASE=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-2-Mini-4bit/snapshots/*/ 2>/dev/null | head -1)
if [ -n "${BASE}" ]; then
    MODEL_MAP="${MODEL_MAP%,},\"inkling-base\":\"${BASE}\"}"
fi

nohup uv run python scripts/pi_shim.py \
    --upstream "http://127.0.0.1:${MLX_PORT}" \
    --port "${SHIM_PORT}" \
    --model-map "${MODEL_MAP}" \
    > logs/shim.log 2>&1 &

sleep 2
echo "✅ Shim ready"
echo ""
echo "✨ All services up. Benchmark: uv run python scripts/bench.py"
echo "   Use:   pi --provider mlx-local --model inkling-unbound-v2"
echo "   Stop:  pkill -f mlx_lm.server && pkill -f pi_shim.py"
