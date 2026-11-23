#!/usr/bin/env python3
"""OpenAI-compatible shim for Inkling-2-Mini (MLX).

Raw model protocol (captured empirically):

    <|message_model|><|content_thinking|>...thinking...<|end_message|>
    <|message_model|><|content_text|>...answer...<|end_message|>
    <|message_model|>bash<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>

This shim:
  - streams reasoning PROGRESSIVELY as delta.reasoning (pi shows thinking live)
  - never leaks tool-call JSON or control tokens into content
  - emits OpenAI tool_calls + finish_reason "tool_calls"
  - maps :thinking suffixes and reasoning_effort to chat_template_kwargs.reasoning_effort

    mlx_lm.server (raw) :8080   <-  pi_shim :8090  <-  pi
"""
import argparse, json, re, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import uuid

SPECIAL = re.compile(r"<\|[a-zA-Z_]+\|>")
CONTENT_MARK = "<|content_text|>"
THINKING_MARK = "<|content_thinking|>"
TOOL_MARK = "<|content_invoke_tool_json|>"
MSG_MARK = "<|message_model|>"
END_MSG = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"

# All control tokens that can appear in the raw stream, longest first matters
# only for overlapping prefixes (they don't overlap, but keep list explicit).
CTRLS = [THINKING_MARK, CONTENT_MARK, TOOL_MARK, MSG_MARK, END_MSG, END_SAMPLING]

# Hold back the tail so a control token split across two deltas survives.
KEEP_TAIL = 32

EFFORTS = {"none": "none", "minimal": "minimal", "low": "low",
           "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}

UPSTREAM = "http://127.0.0.1:8080"
MODEL_ID = ""
MODEL_MAP = {}


def find_ctrl(s):
    """Earliest control token in s -> (index, token) or (None, None)."""
    best_i, best_t = None, None
    for t in CTRLS:
        i = s.find(t)
        if i != -1 and (best_i is None or i < best_i):
            best_i, best_t = i, t
    return best_i, best_t


def extract_tool_calls(text):
    """Pull <|content_invoke_tool_json|>{...}<|end_message|> blocks out."""
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
        rest = rest.replace(END_MSG, "", 1)
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
    return calls, rest


def split_raw(text):
    """raw model text -> (reasoning, content)."""
    content = ""
    think = ""
    if CONTENT_MARK in text:
        head, content = text.rsplit(CONTENT_MARK, 1)
        content = content.split(END_MSG)[0]
        think = "".join(re.findall(re.escape(THINKING_MARK) + r"(.*?)(?=" + re.escape(END_MSG) + r"|$)", head, re.S))
    else:
        think = "".join(re.findall(re.escape(THINKING_MARK) + r"(.*?)(?=" + re.escape(END_MSG) + r"|$)", text, re.S))
        if not think:
            content = SPECIAL.sub("", text)
    return SPECIAL.sub("", think).strip(), SPECIAL.sub("", content).strip()


def forward(path, payload):
    req = urllib.request.Request(
        f"{UPSTREAM}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=900)


class StreamSplitter:
    """Routes raw deltas into reasoning/content/tool streams.

    Modes:
      start    - before first control token
      thinking - text is reasoning, emit as delta.reasoning
      content  - text is answer, emit as delta.content
      tool     - tool call in progress; suppress text (emitted as tool_calls at end)
      drop     - between sections (e.g. fn name after <|message_model|>)
    """

    def __init__(self):
        self.mode = "start"
        self.buf = ""       # pending text not yet routed (current section)
        self.all_raw = ""   # full raw stream, for tool-call extraction at finish

    def feed(self, delta):
        """Returns list of ('reasoning'|'content', text) events."""
        self.all_raw += delta
        events = []
        if self.mode == "tool":
            # tool JSON is opaque: accumulate raw, no scanning, no routing
            self.buf += delta
            return events
        self.buf += delta
        while True:
            i, tok = find_ctrl(self.buf)
            if tok is None:
                break
            before, self.buf = self.buf[:i], self.buf[i + len(tok):]
            self._route(before, events)
            if tok == THINKING_MARK:
                self.mode = "thinking"
            elif tok == CONTENT_MARK:
                self.mode = "content"
            elif tok == TOOL_MARK:
                self.mode = "tool"
                self.buf = TOOL_MARK + self.buf  # keep marker for extraction
                break  # CRITICAL: exit scan, else re-finds TOOL_MARK forever
            elif tok in (MSG_MARK, END_MSG, END_SAMPLING):
                self.mode = "drop"
        if self.mode != "tool" and len(self.buf) > KEEP_TAIL:
            emit, self.buf = self.buf[:-KEEP_TAIL], self.buf[-KEEP_TAIL:]
            self._route(emit, events)
        return events

    def _route(self, text, events):
        if not text:
            return
        if self.mode == "thinking":
            t = SPECIAL.sub("", text)
            if t:
                events.append(("reasoning", t))
        elif self.mode == "content":
            t = SPECIAL.sub("", text)
            if t:
                events.append(("content", t))
        # start/tool/drop: discard

    def finish(self):
        """Flush the tail. Returns (events, tool_calls)."""
        events = []
        if self.mode != "tool":
            self._route(self.buf, events)
            calls = []
        else:
            calls, _ = extract_tool_calls(self.buf)
            if not calls:
                calls, _ = extract_tool_calls(self.all_raw)
        self.buf = ""
        return events, calls


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args): pass

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_response(404); self.end_headers(); return

        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")

        # ---- model id + :thinking suffix -------------------------------
        requested = payload.get("model", "")
        thinking_level = None
        if ":" in requested:
            parts = requested.rsplit(":", 1)
            if parts[1].lower() in EFFORTS:
                thinking_level = parts[1].lower()
                requested = parts[0]
        if requested in MODEL_MAP:
            payload["model"] = MODEL_MAP[requested]
        elif MODEL_ID:
            payload["model"] = MODEL_ID

        # ---- reasoning effort -> chat template kwarg --------------------
        effort = payload.get("reasoning_effort")
        if isinstance(payload.get("reasoning"), dict):
            effort = payload["reasoning"].get("effort") or effort
        payload.pop("reasoning_effort", None)
        kwargs = dict(payload.get("chat_template_kwargs") or {})
        if thinking_level:
            kwargs["reasoning_effort"] = EFFORTS[thinking_level]
        else:
            kwargs.setdefault("reasoning_effort", EFFORTS.get(str(effort).lower(), "low"))
        payload["chat_template_kwargs"] = kwargs

        stream = bool(payload.pop("stream", False))
        payload["stream"] = stream
        payload.setdefault("max_tokens", 4096)

        try:
            resp = forward("/v1/chat/completions", payload)
        except urllib.error.HTTPError as e:
            return self.send_error(e.code, e.read().decode()[:400])

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        try:
            if not stream:
                data = json.loads(resp.read())
                for ch in data.get("choices", []):
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
                return

            # ---- streaming ------------------------------------------------
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            sp = StreamSplitter()
            finish = None
            try:
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if not data.get("choices"):
                        continue
                    ch = data["choices"][0]
                    delta = ch.get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        for kind, text in sp.feed(piece):
                            self._sse({"id": chunk_id, "object": "chat.completion.chunk",
                                       "model": requested,
                                       "choices": [{"index": 0,
                                                    "delta": ({ "reasoning": text } if kind == "reasoning"
                                                              else { "content": text }),
                                                    "finish_reason": None}]})
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                        break
            except (BrokenPipeError, ConnectionResetError):
                return  # client went away (pi interrupted); die quietly
            except Exception as e:
                try:
                    self._sse({"error": str(e)})
                except (BrokenPipeError, ConnectionResetError):
                    return

            events, calls = sp.finish()
            for kind, text in events:
                self._sse({"id": chunk_id, "object": "chat.completion.chunk",
                           "model": requested,
                           "choices": [{"index": 0,
                                        "delta": ({ "reasoning": text } if kind == "reasoning"
                                                  else { "content": text }),
                                        "finish_reason": None}]})
            if calls:
                self._sse({"id": chunk_id, "object": "chat.completion.chunk",
                           "model": requested,
                           "choices": [{"index": 0,
                                        "delta": {"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]},
                                        "finish_reason": None}]})
                finish = "tool_calls"
            self._sse({"id": chunk_id, "object": "chat.completion.chunk",
                       "model": requested,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": finish or "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            try:
                resp.close()  # free the upstream socket even on early break
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model-map", required=True)
    args = parser.parse_args()

    global UPSTREAM, MODEL_MAP
    UPSTREAM = args.upstream
    MODEL_MAP = json.loads(args.model_map)

    print(f"shim on :{args.port} -> {UPSTREAM}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
