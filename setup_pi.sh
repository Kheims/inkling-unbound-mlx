#!/bin/bash
# Configure pi for Inkling-2-Mini Unbound v2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${SCRIPT_DIR}/models/mlx-unbound-v2"
SHIM_PORT=8090
MLX_PORT=8080

echo "🔧 Configuring pi..."

# Update ~/.pi/agent/models.json
PI_CONFIG="$HOME/.pi/agent/models.json"

if [ ! -f "${PI_CONFIG}" ]; then
    echo "❌ ~/.pi/agent/models.json not found. Install pi first."
    exit 1
fi

echo "📝 Updating ${PI_CONFIG}..."

python3 << PYEOF
import json
import sys

config_path = "${PI_CONFIG}"
model_dir = "${MODEL_DIR}"
mlx_port = ${MLX_PORT}
shim_port = ${SHIM_PORT}

try:
    with open(config_path) as f:
        config = json.load(f)
except Exception as e:
    print(f"Error reading config: {e}")
    sys.exit(1)

# Create mlx-local provider if it doesn't exist
if "providers" not in config:
    config["providers"] = {}
if "mlx-local" not in config["providers"]:
    config["providers"]["mlx-local"] = {
        "type": "openai",
        "baseURL": f"http://127.0.0.1:{shim_port}/v1",
        "models": []
    }

# Add models
models = config["providers"]["mlx-local"]["models"]
existing_ids = [m["id"] for m in models]

new_models = [
    {
        "id": "inkling-unbound-v2",
        "name": "Inkling-2-Mini Unbound v2",
        "contextWindow": 100000,
        "maxTokens": 8192,
        "reasoning": True,
        "imageInput": False,
        "toolCalling": True
    },
    {
        "id": "inkling-base",
        "name": "Inkling-2-Mini Base",
        "contextWindow": 100000,
        "maxTokens": 8192,
        "reasoning": True,
        "imageInput": False,
        "toolCalling": True
    }
]

for new_model in new_models:
    if new_model["id"] not in existing_ids:
        models.append(new_model)
        print(f"✅ Added: {new_model['id']}")
    else:
        print(f"ℹ️  Already present: {new_model['id']}")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("✅ Configuration updated")
PYEOF

echo ""
echo "📋 Next steps:"
echo "   1. Start MLX server: ./start_server.sh"
echo "   2. Start shim: python scripts/pi_shim.py --port ${SHIM_PORT} --model-map '{\"inkling-unbound-v2\":\"${MODEL_DIR}\",\"inkling-base\":\"...\"}'"
echo "   3. Use pi: pi --provider mlx-local --model inkling-unbound-v2 \"Your question\""
echo ""
echo "💡 Tip: Create an alias in ~/.zshrc:"
echo "   alias pi-inkling='pi --provider mlx-local --model inkling-unbound-v2'"
