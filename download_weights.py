#!/usr/bin/env python3
"""Download and apply the MLX delta for Inkling-2-Mini Unbound v2."""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    import json
except ImportError as e:
    print(f"Error: {e}")
    print("Install dependencies: pip install huggingface_hub")
    sys.exit(1)

# Config
DELTA_REPO = "Djam1998/Inkling-2-Mini-Unbound-v2-MLX-4bit"
BASE_MODEL = "mlx-community/Inkling-2-Mini-4bit"
OUTPUT_DIR = Path("models/mlx-unbound-v2")

def main():
    print(f"📥 Download du delta depuis {DELTA_REPO}...")
    delta_dir = snapshot_download(repo_id=DELTA_REPO, local_dir="models/delta")
    print(f"✅ Delta téléchargé dans {delta_dir}")
    
    print(f"📥 Download du modèle base {BASE_MODEL}...")
    base_dir = snapshot_download(repo_id=BASE_MODEL, local_dir="models/base")
    print(f"✅ Modèle base téléchargé dans {base_dir}")
    
    print("🔄 Application du delta...")
    apply_script = Path(delta_dir) / "apply.py"
    if apply_script.exists():
        import subprocess
        subprocess.run([sys.executable, str(apply_script), str(base_dir), str(OUTPUT_DIR)], check=True)
        print(f"✅ Delta appliqué dans {OUTPUT_DIR}")
    else:
        # Fallback: copy delta files directly
        import shutil
        shutil.copytree(delta_dir, OUTPUT_DIR, dirs_exist_ok=True)
        print(f"✅ Fichiers copiés dans {OUTPUT_DIR}")
    
    print("\n✅ Prêt ! Lance le serveur avec : ./start_server.sh")

if __name__ == "__main__":
    main()
