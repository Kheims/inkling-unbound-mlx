#!/usr/bin/env python3
"""Benchmark prefill (TTFT) and decode speed for the Inkling MLX server.

Measures, per prompt size:
  - cold TTFT   : time to first token (unique prompt -> full prefill)
  - warm TTFT   : same prompt again (prompt-cache hit)
  - decode speed: tokens/s while generating

Usage:
    uv run python scripts/bench.py                    # defaults, via shim :8090
    uv run python scripts/bench.py --url http://127.0.0.1:8080   # direct, raw protocol
    uv run python scripts/bench.py --sizes 128,2048 --decode-tokens 128

Note: --url pointed at the shim (:8090) benchmarks the full pi path including
thinking-split; pointed at mlx_lm.server (:8080) it benchmarks raw generation.
"""
import argparse
import json
import time
import urllib.request

WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
         "hello", "world", "code", "data", "test", "model", "token"]


def make_prompt(approx_tokens: int, seed: int) -> str:
    """Deterministic pseudo-random word soup (~1 token per word)."""
    out, i = [], seed
    while len(out) < approx_tokens:
        out.append(WORDS[(i * 2654435761) % len(WORDS)])
        i += 1
    return "Count the words in this text: " + " ".join(out)


def stream_once(url: str, model: str, prompt: str, max_tokens: int, effort: str):
    """Returns (ttft_s, decode_tok_s, n_tokens, n_reasoning_chars)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    ttft = None
    n_words = 0          # ~1 token per whitespace-separated word (content + reasoning)
    n_reason_chars = 0
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            ch = d.get("choices") or [{}]
            delta = ch[0].get("delta", {}) if ch else {}
            piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_reason_chars += len(delta.get("reasoning") or "")
                n_words += len(piece.split())
            if ch and ch[0].get("finish_reason"):
                break
    total = time.perf_counter() - t0
    decode_toks = max(1, n_words - 1)   # both thinking and answer are decode tokens
    decode_s = max(1e-9, total - (ttft or 0))
    return ttft, decode_toks / decode_s, n_words, n_reason_chars


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8090",
                   help="shim (:8090) or raw mlx server (:8080)")
    p.add_argument("--model", default="inkling-unbound-v2")
    p.add_argument("--sizes", default="128,1024,4096",
                   help="comma-separated prompt sizes in tokens")
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--effort", default="low",
                   help="reasoning effort for the bench (keep low to time decode, not thinking)")
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"bench -> {args.url}  model={args.model}  effort={args.effort}\n")
    print(f"{'prompt':>8} | {'cold TTFT':>9} {'prefill/s':>9} | {'warm TTFT':>9} | {'decode tok/s':>11}")
    print("-" * 60)

    # warmup: stabilise the server before timing (first call has extra overhead)
    print("warming up...", end="\r")
    stream_once(args.url, args.model, "hi", 8, "none")

    for i, size in enumerate(sizes):
        prompt = make_prompt(size, seed=i)
        # cold: unique prompt (seed varies), measures real prefill
        cold, _, _, _ = stream_once(args.url, args.model, prompt,
                                    args.decode_tokens, args.effort)
        # warm: same prompt again -> prompt-cache hit; also measures decode
        warm, dps, ntok, _ = stream_once(args.url, args.model,
                                          prompt + " Now write a detailed answer.",
                                          args.decode_tokens, args.effort)
        cold_prefill = size / cold if cold else 0
        print(f"{size:>8} | {cold:>8.2f}s {cold_prefill:>8.0f}/s | {warm:>8.2f}s | {dps:>11.1f}")

    print("\nNotes:")
    print("  - cold TTFT includes full prefill of the prompt (+ pi system prompt if via shim)")
    print("  - decode measured on a short continuation; effort=low keeps thinking short")


if __name__ == "__main__":
    main()
