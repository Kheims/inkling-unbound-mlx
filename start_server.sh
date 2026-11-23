#!/bin/bash
# Lancer le serveur MLX pour Inkling-2-Mini Unbound v2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration (optimisée pour M1/M2/M3 Pro)
MODEL_DIR="${SCRIPT_DIR}/models/mlx-unbound-v2"
PORT=8080
HOST="127.0.0.1"

# Pour M5 Pro 64GB, augmente ces valeurs :
PROMPT_CACHE_SIZE=8          # GB de cache de préfixe
PROMPT_CACHE_BYTES=8000000000 # 8GB en bytes
PROMPT_CONCURRENCY=2         # requêtes de préfixe simultanées
DECODE_CONCURRENCY=2         # décodages simultanés

echo "🚀 Démarrage du serveur MLX..."
echo "   Modèle: ${MODEL_DIR}"
echo "   Port: ${PORT}"
echo "   Cache: ${PROMPT_CACHE_SIZE}GB"

# Vérifier que le modèle existe
if [ ! -d "${MODEL_DIR}" ]; then
    echo "❌ Modèle non trouvé dans ${MODEL_DIR}"
    echo "   Lance d'abord : python download_weights.py"
    exit 1
fi

# Lancer le serveur
exec python -m mlx_lm.server \
    --model "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-tokens 8192 \
    --prompt-cache-size "${PROMPT_CACHE_SIZE}" \
    --prompt-cache-bytes "${PROMPT_CACHE_BYTES}" \
    --prompt-concurrency "${PROMPT_CONCURRENCY}" \
    --decode-concurrency "${DECODE_CONCURRENCY}"
