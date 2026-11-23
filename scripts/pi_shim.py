#!/usr/bin/env python3
"""OpenAI-compatible shim for Inkling-2-Mini (MLX).

Handles thinking/content split and tool calling.

    mlx_lm.server  (raw)  :8080   <-  pi_shim :8090  <-  pi

Usage: python scripts/pi_shim.py --upstream http://127.0.0.1:8080 --port 8090
"""
import argparse, json, re, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import uuid

SPECIAL = re.compile(r"<\|[a-zA-Z_]+\|>")
CONTENT_MARK = "<|content_text|>"
TOOL_MARK = "<|content_invoke_tool_json|>"
END_SAMPLING = "<|content_model_end_sampling|>"

UPSTREAM = "http://127.0.0.1:8080"
MODEL_MAP = {}


def extract_tool_calls(text):
    """Extract tool calls from model output."""
    calls, rest = [], text
    while TOOL_MARK in rest:
        head, tail = rest.split(TOOL_MARK, 1)
        i, depth, in_str, esc = 0, 0, False, False
        while i < len(tail):
            c = tail[i]
            if in_str:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': in_str = False
            elif c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        blob, rest = tail[:i], head + tail[i:]
        rest = rest.replace("<|end_message|>", "", 1)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            calls.append({
                "id": f"call_{len(calls)}_{abs(hash(obj['name'])) % 10000}",
                "type": "function",
                "function": {
                    "name": obj["name"],
                    "arguments": json.dumps(obj.get("args", {}), separators=(",", ":"))
                }
            })
    return calls, rest.replace(END_SAMPLING, "")


def split_raw(text):
    """Split raw model text into (reasoning, content)."""
    content = ""
    think = ""
    if CONTENT_MARK in text:
        head, content = text.rsplit(CONTENT_MARK, 1)
        content = content.split("<|end_message|>")[0]
        think = "".join(re.findall(r"<\|content_thinking\|>(.*?)(?=<\|end_message\|>|$)", head, re.S))
    else:
        think = "".join(re.findall(r"<\|content_thinking\|>(.*?)(?=<\|end_message\|>|$)", text, re.S))
        if not think:
            content = SPECIAL.sub("", text)
    return SPECIAL.sub("", think).strip(), SPECIAL.sub("", content).strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args): pass

    def do_POST(self):
        global MODEL_MAP, UPSTREAM

        if self.path != "/v1/chat/completions":
            self.send_response(404); self.end_headers(); return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode())

        model_id = body.get("model", "")
        if model_id in MODEL_MAP:
            body["model"] = MODEL_MAP[model_id]

        # Add reasoning_effort if not present (default to low)
        if "reasoning_effort" not in body:
            body["reasoning_effort"] = "low"

        stream = body.get("stream", False)

        req = urllib.request.Request(
            f"{UPSTREAM}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}
        )

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            chunk_id = str(uuid.uuid4())
            state = {"raw": "", "in_content": False, "content_started": False}

            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    for line in resp:
                        line = line.decode().strip()
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if "choices" in data and data["choices"]:
                                ch = data["choices"][0]
                                delta = ch.get("delta", {})
                                content_delta = delta.get("content", "")

                                if content_delta:
                                    state["raw"] += content_delta
                                    if CONTENT_MARK in state["raw"] and not state["in_content"]:
                                        state["in_content"] = True
                                        state["raw"] = state["raw"].split(CONTENT_MARK, 1)[1]

                                if state["in_content"]:
                                    out = {
                                        "id": chunk_id, "object": "chat.completion.chunk",
                                        "model": body.get("model", ""),
                                        "choices": [{"index": 0, "delta": {"content": content_delta}, "finish_reason": None}]
                                    }
                                    self.wfile.write(f"data: {json.dumps(out)}\n\n".encode())
                                    self.wfile.flush()

                                finish = ch.get("finish_reason")
                                if finish:
                                    if TOOL_MARK in state["raw"]:
                                        calls, _ = extract_tool_calls(state["raw"])
                                        if calls:
                                            out = {
                                                "id": chunk_id, "object": "chat.completion.chunk",
                                                "model": body.get("model", ""),
                                                "choices": [{"index": 0, "delta": {"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]}, "finish_reason": None}]
                                            }
                                            self.wfile.write(f"data: {json.dumps(out)}\n\n".encode())
                                            self.wfile.flush()
                                            finish = "tool_calls"

                                    out = {
                                        "id": chunk_id, "object": "chat.completion.chunk",
                                        "model": body.get("model", ""),
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
                                    }
                                    self.wfile.write(f"data: {json.dumps(out)}\n\n".encode())
                                    self.wfile.write(b"data: [DONE]\n\n")
                                    self.wfile.flush()
                                    break
            except Exception as e:
                self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
        else:
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    data = json.load(resp)
                    if "choices" in data and data["choices"]:
                        ch = data["choices"][0]
                        m = ch.get("message", {})
                        raw = m.get("content", "")
                        calls, raw_body = extract_tool_calls(raw)
                        r, c = split_raw(raw_body)
                        m["content"] = c
                        if r:
                            m["reasoning"] = r
                        if calls:
                            m["tool_calls"] = calls
                            ch["finish_reason"] = "tool_calls"
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode())
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(e.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model-map", required=True)
    args = parser.parse_args()

    global UPSTREAM, MODEL_MAP
    UPSTREAM = args.upstream
    MODEL_MAP = json.loads(args.model_map)

    print(f"Shim listening on port {args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
