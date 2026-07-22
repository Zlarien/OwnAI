"""A tiny local web chat UI, built on http.server and nothing else.

The interesting parts are deliberately decoupled from the socket so they can be
unit-tested without ever binding a port:

  * ``PAGE_HTML``       — the whole single-page app (inline CSS + vanilla JS).
  * ``answer_to_json``  — turn an answerer + question into a plain JSON dict.
  * ``build_handler``   — a request-handler class closed over a live answerer.
  * ``serve``           — the only function that actually opens a server.

Keeping the logic outside the handler means the tests exercise real behaviour
with an in-memory retriever, and only ``serve`` touches the network.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# A self-contained ChatGPT-like page: no external fonts, no CDN, no build step.
# Theme follows the OS via prefers-color-scheme so it looks native in light or
# dark. All CSS and JS are inline so a single GET serves the entire app.
PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OwnAI</title>
<style>
  :root {
    --bg: #ffffff;
    --panel: #f7f7f8;
    --text: #1f2023;
    --muted: #6b7280;
    --border: #e5e7eb;
    --user-bg: #2563eb;
    --user-text: #ffffff;
    --ai-bg: #f0f1f3;
    --ai-text: #1f2023;
    --accent: #2563eb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1a1b1e;
      --panel: #202124;
      --text: #e8eaed;
      --muted: #9aa0a6;
      --border: #34363a;
      --user-bg: #2563eb;
      --user-text: #ffffff;
      --ai-bg: #2a2c30;
      --ai-text: #e8eaed;
      --accent: #4f8cff;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    font-size: 16px;
    background: var(--panel);
  }
  header .sub { font-weight: 400; color: var(--muted); font-size: 13px; }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .row { display: flex; }
  .row.user { justify-content: flex-end; }
  .row.ai { justify-content: flex-start; }
  .bubble {
    max-width: min(680px, 85%);
    padding: 10px 14px;
    border-radius: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .user .bubble { background: var(--user-bg); color: var(--user-text); border-bottom-right-radius: 4px; }
  .ai .bubble { background: var(--ai-bg); color: var(--ai-text); border-bottom-left-radius: 4px; }
  .sources { margin-top: 6px; font-size: 12px; color: var(--muted); }
  form {
    display: flex;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    background: var(--panel);
  }
  #q {
    flex: 1;
    padding: 11px 14px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg);
    color: var(--text);
    font-size: 15px;
    outline: none;
  }
  #q:focus { border-color: var(--accent); }
  button {
    padding: 0 18px;
    border: none;
    border-radius: 12px;
    background: var(--accent);
    color: #fff;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
  <header>OwnAI <span class="sub">— domain chat, 100% from scratch</span></header>
  <div id="messages"></div>
  <form id="chat">
    <input id="q" type="text" autocomplete="off"
           placeholder="Ask something about the domain..." />
    <button type="submit" id="send">Send</button>
  </form>
<script>
  var messages = document.getElementById("messages");
  var form = document.getElementById("chat");
  var input = document.getElementById("q");
  var send = document.getElementById("send");

  function addBubble(role, text, sources) {
    var row = document.createElement("div");
    row.className = "row " + role;
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    if (sources && sources.length) {
      var s = document.createElement("div");
      s.className = "sources";
      s.textContent = "sources: " + sources.join("; ");
      bubble.appendChild(s);
    }
    row.appendChild(bubble);
    messages.appendChild(row);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var question = input.value.trim();
    if (!question) return;
    addBubble("user", question);
    input.value = "";
    send.disabled = true;
    var pending = addBubble("ai", "...");
    fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question })
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        pending.textContent = data.answer || "";
        if (data.sources && data.sources.length) {
          var s = document.createElement("div");
          s.className = "sources";
          s.textContent = "sources: " + data.sources.join("; ");
          pending.appendChild(s);
        }
      })
      .catch(function () { pending.textContent = "(error contacting the server)"; })
      .finally(function () {
        send.disabled = false;
        input.focus();
        messages.scrollTop = messages.scrollHeight;
      });
  });
  input.focus();
</script>
</body>
</html>
"""


def answer_to_json(answerer, question: str, top_k: int = 4) -> dict:
    """Run one question through an answerer and return a JSON-ready dict.

    Uses the existing ``Answer`` dataclass fields (``.text`` and ``.sources``)
    so both the extractive and generative answerers work unchanged.
    """
    ans = answerer.answer(question, top_k=top_k)
    return {"answer": ans.text, "sources": list(ans.sources)}


def build_handler(answerer, top_k: int = 4):
    """Return a BaseHTTPRequestHandler subclass closed over a live answerer.

    Factored out of ``serve`` so tests can assert on the class without ever
    starting a server: GET / serves the page, POST /ask answers a question.
    """

    class ChatHandler(BaseHTTPRequestHandler):
        # Quiet by default; the console already prints the served URL.
        def log_message(self, *args):  # noqa: D401 - silence default logging
            pass

        def _send(self, code, content_type, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.split("?")[0] == "/":
                self._send(200, "text/html; charset=utf-8", PAGE_HTML.encode("utf-8"))
            else:
                self._send(404, "text/plain; charset=utf-8", b"Not Found")

        def do_POST(self):
            if self.path.split("?")[0] != "/ask":
                self._send(404, "text/plain; charset=utf-8", b"Not Found")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                question = (payload.get("question") or "").strip()
            except (ValueError, AttributeError):
                self._send(400, "application/json; charset=utf-8",
                           json.dumps({"answer": "Invalid request.", "sources": []}).encode("utf-8"))
                return
            result = answer_to_json(answerer, question, top_k=top_k)
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(result).encode("utf-8"))

    return ChatHandler


def _artifacts_name(cfg, domain_path) -> str:
    from pathlib import Path

    return cfg.get("name") or Path(domain_path).stem


def serve(domain_path, host: str = "127.0.0.1", port: int = 8000, generative: bool = False):
    """Load a domain and serve the chat UI locally until interrupted."""
    from pathlib import Path

    from ownai.pipeline import load_domain_config
    from ownai.rag import ExtractiveAnswerer, GenerativeAnswerer
    from ownai.retrieval import HybridRetriever

    cfg = load_domain_config(domain_path)
    name = _artifacts_name(cfg, domain_path)
    art = Path("artifacts") / name

    retr = HybridRetriever.load(art / "index")
    if generative:
        from ownai.model import GPT
        from ownai.tokenizer import BPETokenizer

        model = GPT.load(art / "model.npz")
        tok = BPETokenizer.load(art / "tokenizer.json")
        answerer = GenerativeAnswerer(retr, model, tok)
    else:
        answerer = ExtractiveAnswerer(retr)

    top_k = cfg.get("retrieval", {}).get("top_k", 4)
    handler = build_handler(answerer, top_k=top_k)
    httpd = HTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"OwnAI web chat serving at {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        httpd.server_close()
