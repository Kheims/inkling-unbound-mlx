# Inkling-2-Mini Unbound v2 (MLX 4-bit)

**Abliterated/uncensored** version of the Inkling-2-Mini model, optimized for Apple Silicon (MLX).
Runs locally, works as a coding agent with [pi](https://github.com/earendil-works/pi-coding-agent) — thinking traces, tool calling (bash/read/write/edit), 100K context.

## 🚀 Quick Start

```bash
git clone https://github.com/Kheims/inkling-unbound-mlx.git
cd inkling-unbound-mlx

# 1. uv (if not installed) — https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. deps + weights (~350 MB delta on top of the ~17 GB base model)
uv sync
uv run python download_weights.py

# 3. pi (coding agent)
npm install -g @earendil-works/pi-coding-agent
./setup_pi.sh

# 4. serve (MLX server + pi shim)
./start_all.sh

# 5. use
pi --provider mlx-local --model inkling-unbound-v2 "Your question"
```

## 📊 Specifications

| Parameter | Value |
|-----------|-------|
| Model | Inkling-2-Mini Unbound v2 (abliterated) |
| Quantization | 4-bit (MLX) |
| Size | ~17 GB |
| Context | **100K** usable (model natively supports 1M) |
| Max output | 8 192 tokens |
| Prefill (M1 Pro) | ~1 400–6 000 tok/s (hybrid sliding-window architecture) |
| Decode (M1 Pro) | ~21–23 tok/s |
| Warm TTFT | < 1 s (prompt cache) |

## 🧠 Reasoning Control

The model always shows its thinking. Control the depth with a `:level` suffix:

```bash
pi --model mlx-local/inkling-unbound-v2:low      # short thinking, fast
pi --model mlx-local/inkling-unbound-v2:high      # deep thinking (default: high)
```

Levels: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`
(effort map in the chat template: none=0.0, minimal=0.1, low=0.2, medium=0.7, high=0.9, xhigh/max=0.99)

## 🌐 Web Search

`setup_pi.sh` installs a **websearch** pi extension (no API key, DuckDuckGo Lite).
The model gets two extra tools automatically — no system prompt change needed, pi
injects tool schemas into every request:

- `websearch` — search the web, returns top results (title, URL, snippet)
- `webfetch` — fetch a URL and read its text content

```bash
pi --model mlx-local/inkling-unbound-v2 "What's the latest mlx-lm release? Search the web."
```

Source: [`extensions/websearch.ts`](extensions/websearch.ts)

## 📈 Benchmark

```bash
uv run python scripts/bench.py                       # via shim (full pi path)
uv run python scripts/bench.py --url http://127.0.0.1:8080   # raw MLX server
uv run python scripts/bench.py --sizes 128,4096 --decode-tokens 128
```

Measures cold prefill (TTFT), warm-cache TTFT, and decode tok/s per prompt size.

## 🔧 Server Configuration

`start_all.sh` defaults are tuned for **32 GB** machines (17 GB weights leave little headroom):
`--prompt-cache-size 4`, `--prompt-concurrency 1`, `--decode-concurrency 2`.

**64 GB machines** (M5 Pro etc.): raise to `--prompt-cache-size 16`, `--prompt-concurrency 2`.
Too-large caches cause GPU OOM (`generation thread died`) on 32 GB — if you see it, lower the cache.

## 🛠️ Tools

Tool calling works end-to-end in pi (bash, read, write, edit):

```bash
pi --provider mlx-local --model inkling-unbound-v2 \
  "Use bash to list files and count them."
```

## 📦 Structure

```
inkling-unbound-mlx/
├── pyproject.toml         # uv project (mlx, mlx-lm, huggingface_hub)
├── download_weights.py   # fetch delta from HF + apply to base model
├── setup_pi.sh           # register model in ~/.pi/agent/models.json
├── start_all.sh          # serve MLX + shim (one command)
└── scripts/
    ├── pi_shim.py         # thinking split + tool calls (OpenAI-compat)
    └── bench.py           # prefill/decode benchmark
```

## 🐛 Troubleshooting

- **"generation thread died"** — reduce `--prompt-concurrency` / `--decode-concurrency` in `start_all.sh`.
- **First call is slow** — model weights are paged in on first use (~10–30 s); later calls are fast.
- **Ctrl+C during generation** — safe; the shim exits quietly and the socket is freed.

## 📄 License

See the base model: https://huggingface.co/mlx-community/Inkling-2-Mini-4bit
Delta: https://huggingface.co/Djam1998/Inkling-2-Mini-Unbound-v2-MLX-4bit
