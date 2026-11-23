# Inkling-2-Mini Unbound v2 (MLX 4-bit)

**Abliterated/uncensored** version of the Inkling-2-Mini model, optimized for Apple Silicon (MLX).

## 🚀 Quick Start

### Option A: One-command setup (recommended)

```bash
git clone https://github.com/Kheims/inkling-unbound-mlx.git
cd inkling-unbound-mlx
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python download_weights.py
npm install -g @earendil-works/pi-coding-agent
./setup_pi.sh
./start_all.sh
```

Then use:

```bash
pi --provider mlx-local --model inkling-unbound-v2 "Your question"
```

### Option B: Manual setup

#### 1. Clone and install

```bash
git clone https://github.com/Kheims/inkling-unbound-mlx.git
cd inkling-unbound-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. Download weights

```bash
python download_weights.py
```

Downloads the delta from Hugging Face and applies it to the base model.

#### 3. Install pi

```bash
npm install -g @earendil-works/pi-coding-agent
```

#### 4. Configure pi

```bash
./setup_pi.sh
```

Adds `inkling-unbound-v2` to `~/.pi/agent/models.json`.

#### 5. Start services

```bash
./start_all.sh
```

Starts both the MLX server and the pi shim automatically.

#### 6. Use

```bash
pi --provider mlx-local --model inkling-unbound-v2 "Your question"
```

## 📊 Specifications

| Parameter | Value |
|-----------|--------|
| Model | Inkling-2-Mini Unbound v2 |
| Quantization | 4-bit (MLX) |
| Size | ~17 GB |
| Context | 32K tokens |
| Speed (M1 Pro) | ~17 tok/s |
| Speed (M5 Pro 64GB) | ~25+ tok/s |

## 🔧 Configuration

### Server settings (in `start_all.sh`)

- `--prompt-cache-size 8`: 8GB prefix cache (increase to 16 for M5 Pro 64GB)
- `--prompt-cache-bytes 8000000000`: cache memory limit
- `--prompt-concurrency 2`: concurrent prefix requests
- `--decode-concurrency 2`: concurrent decodes

### Reasoning control

The model uses a thinking protocol. By default, `reasoning_effort: low` is set to reduce over-thinking.

To adjust, modify `scripts/pi_shim.py` or pass `reasoning_effort` in your pi config.

## 🛠️ Tools

The model supports tool calling (bash, read, write, edit) via pi.

Example:

```bash
pi --provider mlx-local --model inkling-unbound-v2 \
  "Use bash to list files and count them."
```

## 📦 Structure

```
inkling-unbound-mlx/
├── README.md
├── requirements.txt
├── download_weights.py
├── start_all.sh          # Start everything (server + shim)
├── start_server.sh       # Start MLX server only
├── setup_pi.sh           # Configure pi
├── models/               # MLX weights (after download)
└── scripts/
    └── pi_shim.py        # OpenAI-compatible shim
```

## 🐛 Troubleshooting

### "generation thread died"
Reduce `--prompt-concurrency` and `--decode-concurrency` in `start_all.sh`.

### "model does not support tool calling"
The `pi_shim.py` handles this automatically.

### Slow startup
The cache fills after the first request. Subsequent requests are fast.

### Model not found
Run `python download_weights.py` first.

## 📄 License

See the base model: https://huggingface.co/mlx-community/Inkling-2-Mini-4bit

## 🧠 Reasoning Control

The model supports different reasoning effort levels. Use the `:thinking` suffix in the model ID:

```bash
# No reasoning (fastest, minimal thinking)
pi --model mlx-local/inkling-unbound-v2:none "Quick answer please"

# Minimal reasoning (default, balanced)
pi --model mlx-local/inkling-unbound-v2:minimal "Your question"

# Low reasoning (some thinking)
pi --model mlx-local/inkling-unbound-v2:low "Your question"

# Medium reasoning (more thorough)
pi --model mlx-local/inkling-unbound-v2:medium "Complex problem"

# High reasoning (most thorough, slower)
pi --model mlx-local/inkling-unbound-v2:high "Very complex problem"
```

Available levels: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`

**Default**: `none` (fastest responses)
