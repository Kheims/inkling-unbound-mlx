# Inkling-2-Mini Unbound v2 (MLX 4-bit)

**Abliterated/uncensored** version of the Inkling-2-Mini model, optimized for Apple Silicon (MLX).

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Kheims/inkling-unbound-mlx.git
cd inkling-unbound-mlx
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download weights (delta)

```bash
python download_weights.py
```

This downloads the delta from Hugging Face (`Djam1998/Inkling-2-Mini-Unbound-v2-MLX-4bit`) and applies it to the base model (`mlx-community/Inkling-2-Mini-4bit`).

### 4. Start the MLX server

```bash
./start_server.sh
```

The server listens on `http://127.0.0.1:8080`.

### 5. Install pi (coding agent)

```bash
npm install -g @earendil-works/pi-coding-agent
```

### 6. Configure pi

```bash
./setup_pi.sh
```

This adds the `inkling-unbound-v2` model to `~/.pi/agent/models.json`.

### 7. Use

```bash
pi --provider mlx-local --model inkling-unbound-v2 "Your question here"
```

## 📊 Specifications

| Parameter | Value |
|-----------|--------|
| Model | Inkling-2-Mini Unbound v2 |
| Quantization | 4-bit (MLX) |
| Size | ~17 GB |
| Context | 16K tokens |
| Speed (M1 Pro) | ~17 tok/s |
| Speed (M5 Pro 64GB) | ~25+ tok/s (estimated) |

## 🔧 Server Configuration

The `start_server.sh` script uses:
- `--prompt-cache-size 8`: 8GB prefix cache
- `--prompt-cache-bytes 8000000000`: cache memory limit
- `--prompt-concurrency 2`: 2 concurrent prefix requests
- `--decode-concurrency 2`: 2 concurrent decodes

For M5 Pro 64GB, increase these values in `start_server.sh`.

## 🛠️ Tools

The model supports tool calling (bash, read, write, edit) via pi.

Example:

```bash
pi --provider mlx-local --model inkling-unbound-v2 \
  "Use bash to list files in the current directory and count them."
```

## 📦 Structure

```
inkling-unbound-mlx/
├── README.md
├── requirements.txt
├── download_weights.py
├── start_server.sh
├── setup_pi.sh
├── models/               # MLX weights (after download)
└── scripts/
    └── pi_shim.py        # shim for tool calling support
```

## 🐛 Troubleshooting

### "generation thread died"
Reduce `--prompt-concurrency` and `--decode-concurrency` in `start_server.sh`.

### "model does not support tool calling"
The `pi_shim.py` shim handles this automatically.

### Slow startup
The cache fills after the first request. Subsequent requests are fast.

## 📄 License

See the base model: https://huggingface.co/mlx-community/Inkling-2-Mini-4bit
