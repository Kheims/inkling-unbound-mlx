#!/bin/bash
# Configurer pi pour utiliser Inkling-2-Mini Unbound v2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${SCRIPT_DIR}/models/mlx-unbound-v2"
SHIM_PORT=8090
MLX_PORT=8080

echo "🔧 Configuration de pi..."

# Créer le shim si nécessaire
if [ ! -f "scripts/pi_shim.py" ]; then
    mkdir -p scripts
    echo "⚠️  scripts/pi_shim.py manquant. Copier depuis le repo original."
    exit 1
fi

# Mettre à jour ~/.pi/agent/models.json
PI_CONFIG="$HOME/.pi/agent/models.json"

if [ ! -f "${PI_CONFIG}" ]; then
    echo "❌ ~/.pi/agent/models.json non trouvé. Installe d'abord pi."
    exit 1
fi

echo "📝 Mise à jour de ${PI_CONFIG}..."

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
    print(f"Erreur lecture config: {e}")
    sys.exit(1)

# Créer le provider mlx-local s'il n'existe pas
if "providers" not in config:
    config["providers"] = {}
if "mlx-local" not in config["providers"]:
    config["providers"]["mlx-local"] = {
        "type": "openai",
        "baseURL": f"http://127.0.0.1:{shim_port}/v1",
        "models": []
    }

# Ajouter les modèles
models = config["providers"]["mlx-local"]["models"]
existing_ids = [m["id"] for m in models]

new_models = [
    {
        "id": "inkling-unbound-v2",
        "name": "Inkling-2-Mini Unbound v2",
        "contextWindow": 32768,
        "maxTokens": 8192,
        "reasoning": True,
        "imageInput": False,
        "toolCalling": True
    },
    {
        "id": "inkling-base",
        "name": "Inkling-2-Mini Base",
        "contextWindow": 32768,
        "maxTokens": 8192,
        "reasoning": True,
        "imageInput": False,
        "toolCalling": True
    }
]

for new_model in new_models:
    if new_model["id"] not in existing_ids:
        models.append(new_model)
        print(f"✅ Ajouté: {new_model['id']}")
    else:
        print(f"ℹ️  Déjà présent: {new_model['id']}")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("✅ Configuration mise à jour")
PYEOF

echo ""
echo "📋 Prochaines étapes :"
echo "   1. Lance le serveur MLX : ./start_server.sh"
echo "   2. Lance le shim : python scripts/pi_shim.py --port ${SHIM_PORT} --model-map '{\"inkling-unbound-v2\":\"${MODEL_DIR}\",\"inkling-base\":\"...\"}'"
echo "   3. Utilise pi : pi --provider mlx-local --model inkling-unbound-v2 \"Ta question\""
echo ""
echo "💡 Astuce : crée un alias dans ton ~/.zshrc :"
echo "   alias pi-inkling='pi --provider mlx-local --model inkling-unbound-v2'"
