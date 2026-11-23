#!/usr/bin/env python3
"""OpenAI-compatible shim in front of `mlx_lm.server` for Inkling-2-Mini.

mlx-lm does not understand this model's thinking protocol (it emits raw
<|message_model|><|content_thinking|>...<|content_text|>... tokens), so it puts
the reasoning AND the control tokens into `content`. This proxy rewrites it into
a proper OpenAI payload: `reasoning` + clean `content`, streaming included.

    mlx_lm.server  (raw)  :8080   <-  pi_shim :8090  <-  pi

Usage: python scripts/pi_shim.py --upstream http://127.0.0.1:8080 --port 8090
"""
import argparse, json, re, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SPECIAL = re.compile(r"<\|[a-zA-Z_]+\|>")
CONTENT_MARK = "<|content_text|>"
EFFORTS = {"none": "none", "minimal": "minimal", "low": "low",
           "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}
KEEP_TAIL = 24  # chars held back so a control token split across deltas survives

UPSTREAM = "http://127.0.0.1:8080"
MODEL_ID = ""   # upstream model id; requests are rewritten to it
MODEL_MAP = {}  # pi-facing model id -> upstream model id


TOOL_MARK = "<|content_invoke_tool_json|>"
END_SAMPLING = "<|content_model_end_sampling|>"


def extract_tool_calls(text):
    """Pull <|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|> blocks out."""
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
            calls.append({"id": f"call_{len(calls)}_{abs(hash(obj['name'])) % 10000}",
                          "type": "function",
                          "function": {"name": obj["name"],
                                       "arguments": json.dumps(obj.get("args", {}),
                                                               separators=(",", ":"))}})
    return calls, rest.replace(END_SAMPLING, "")


def split_raw(text):
    """raw model text -> (reasoning, content)."""
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


def _trim_partial(text):
    """Drop a trailing, not-yet-closed control marker (e.g. '<|message_m')."""
    i = text.rfind("<|")
    if i != -1 and "|>" not in text[i:]:
        return text[:i]
    return text


class Streamer:
    """Turns raw model deltas into (reasoning_delta, content_delta) pairs."""

    def __init__(self):
        self.raw = ""
        self.in_content = False
        self.reasoning_done = False
        self.emitted_reasoning = 0
        self.emitted_content = 0

    def feed(self, delta, final=False):
        self.raw += delta
        r_out = c_out = ""

        if not self.in_content and CONTENT_MARK in self.raw:
            self.in_content = True

        if not self.in_content:
            head = self.raw.rsplit("<|content_thinking|>", 1)[-1]
            head = SPECIAL.sub("", _trim_partial(head))
            if not final:
                head = head[:-16] if len(head) > 16 else ""
            if len(head) > self.emitted_reasoning:
                r_out = head[self.emitted_reasoning:]
                self.emitted_reasoning = len(head)
            return r_out, c_out

        # content phase: first, flush whatever reasoning was still pending
        if not self.reasoning_done:
            self.reasoning_done = True
            head = self.raw.split(CONTENT_MARK, 1)[0]
            head = SPECIAL.sub("", head)
            if len(head) > self.emitted_reasoning:
                r_out = head[self.emitted_reasoning:].lstrip()
                self.emitted_reasoning = len(head)

        body = self.raw.split(CONTENT_MARK, 1)[1]
        body = body.split("<|end_message|>")[0]
        body = SPECIAL.sub("", _trim_partial(body))
        if len(body) > self.emitted_content:
            c_out = body[self.emitted_content:]
            self.emitted_content = len(body)
        return r_out, c_out


def forward(path, payload):
    req = urllib.request.Request(UPSTREAM + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0: the SSE body is terminated by connection close, which every
    # OpenAI client understands (with 1.1 + no chunking, pi retried the request).
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *a):
        print("  " + fmt % a, flush=True)

    def do_GET(self):
        try:
            with forward(self.path.rstrip("/"), None) if False else urllib.request.urlopen(
                    UPSTREAM + self.path, timeout=30) as r:
                body = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_error(502, str(e))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._passthrough(payload)

        effort = payload.get("reasoning_effort") or payload.get("reasoning", {}).get("effort") \
            if isinstance(payload.get("reasoning"), dict) else payload.get("reasoning_effort")
        payload.pop("reasoning_effort", None)
        kwargs = dict(payload.get("chat_template_kwargs") or {})
        kwargs.setdefault("reasoning_effort", EFFORTS.get(str(effort).lower(), "low"))
        payload["chat_template_kwargs"] = kwargs

        requested = payload.get("model", "")
        if requested in MODEL_MAP:
            payload["model"] = MODEL_MAP[requested]
        elif MODEL_ID:
            payload["model"] = MODEL_ID
        stream = bool(payload.pop("stream", False))
        payload["stream"] = stream
        payload.setdefault("max_tokens", 4096)

        try:
            resp = forward("/v1/chat/completions", payload)
        except urllib.error.HTTPError as e:
            return self.send_error(e.code, e.read().decode()[:400])

        if not stream:
            data = json.loads(resp.read())
            for ch in data.get("choices", []):
                m = ch.get("message") or {}
                calls, raw_body = extract_tool_calls(m.get("content") or "")
                r, c = split_raw(raw_body)
                m["content"] = c
                if r:
                    m["reasoning"] = r
                if calls:
                    m["tool_calls"] = calls
                    ch["finish_reason"] = "tool_calls"
                if ch.get("finish_reason") == "length" and CONTENT_MARK not in (m.get("content") or "") and not c:
                    ch["finish_reason"] = "length"
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        MODEL = payload.get("model") or "inkling-unbound-v2"
        state = Streamer()
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            raw = delta.get("content") or ""
            finish = choice.get("finish_reason")
            r, c = state.feed(raw, final=bool(finish))
            for kind, txt in (("reasoning", r), ("content", c)):
                if txt:
                    out = {"id": obj.get("id", "chatcmpl-shim"), "object": "chat.completion.chunk",
                           "model": MODEL, "choices": [{"index": 0, "delta": {kind: txt},
                                                          "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(out)}\n\n".encode())
                    self.wfile.flush()
            if finish:
                if TOOL_MARK in state.raw or state.in_content:
                    calls, _ = extract_tool_calls(state.raw)
                    if calls:
                        delta = {"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]}
                        out = {"id": obj.get("id", "chatcmpl-shim"),
                               "object": "chat.completion.chunk", "model": MODEL,
                               "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(out)}\n\n".encode())
                        self.wfile.flush()
                        finish = "tool_calls"
                out = {"id": obj.get("id", "chatcmpl-shim"), "object": "chat.completion.chunk",
                       "model": MODEL, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(out)}\n\n".encode())
                self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _passthrough(self, payload):
        try:
            with forward(self.path, payload) as r:
                body = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except urllib.error.HTTPError as e:
            self.send_error(e.code, e.read().decode()[:400])


def main():
    global UPSTREAM, MODEL_ID, MODEL_MAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model-id", default="", help="upstream model id every request is rewritten to")
    ap.add_argument("--model-map", default="{}", help='JSON map: pi model id -> upstream model id')
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    MODEL_ID = args.model_id
    MODEL_MAP = json.loads(args.model_map)
    print(f"shim on :{args.port} -> {UPSTREAM} (model={MODEL_ID or 'as-requested'})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
