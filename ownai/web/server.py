"""A tiny local web chat UI, built on http.server and nothing else.

The interesting parts are deliberately decoupled from the socket so they can be
unit-tested without ever binding a port:

  * ``PAGE_HTML``       : the whole single-page app (inline CSS + vanilla JS).
  * ``answer_to_json``  : turn an answerer + question into a plain JSON dict.
  * ``DomainRegistry``  : every built domain on disk, answerers loaded lazily.
  * ``build_handler``   : a request-handler class closed over a registry.
  * ``serve``           : the only function that actually opens a server.

The UI can switch domain and answer mode (extractive, NumPy mini-GPT, OwnGPT)
live, so one server shows the whole stack side by side. OwnGPT replies stream
token by token over ``POST /chat_stream`` and keep the conversation history.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# A self-contained chat page: no external fonts, no CDN, no build step.
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
    --bg: #fbfbfa; --panel: #ffffff; --text: #1c1d1f; --muted: #6b6f76;
    --border: #e4e4e0; --accent: #d9480f; --accent-soft: #fff1e8;
    --user-bg: #1c1d1f; --user-text: #ffffff; --ai-bg: #ffffff; --warn: #a15c00;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #141517; --panel: #1b1c1f; --text: #e9e9e6; --muted: #979aa0;
      --border: #2d2f33; --accent: #ff7a3d; --accent-soft: #2a1d15;
      --user-bg: #e9e9e6; --user-text: #141517; --ai-bg: #1b1c1f; --warn: #f0b458;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg); color: var(--text);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex; flex-direction: column;
  }
  header {
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px;
    padding: 12px 16px; border-bottom: 1px solid var(--border); background: var(--panel);
  }
  .brand { font-weight: 700; font-size: 17px; letter-spacing: -0.01em; }
  .brand small { font-weight: 400; color: var(--muted); font-size: 12px; margin-left: 6px; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-left: auto; align-items: center; }
  select {
    font: inherit; font-size: 14px; padding: 6px 10px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); max-width: 100%;
  }
  .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .seg button {
    font: inherit; font-size: 13px; padding: 6px 12px; border: 0; background: var(--bg);
    color: var(--muted); cursor: pointer;
  }
  .seg button.on { background: var(--accent); color: #fff; }
  .seg button:disabled { opacity: 0.4; cursor: not-allowed; }
  main { flex: 1; overflow-y: auto; }
  #messages { max-width: 760px; margin: 0 auto; padding: 20px 16px; display: flex; flex-direction: column; gap: 14px; }
  .empty { text-align: center; color: var(--muted); padding: 40px 8px 8px; }
  .empty h1 { color: var(--text); font-size: 22px; margin: 0 0 6px; letter-spacing: -0.02em; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 18px; }
  .chip {
    font: inherit; font-size: 13px; padding: 7px 12px; border-radius: 999px; cursor: pointer;
    border: 1px solid var(--border); background: var(--panel); color: var(--text); text-align: left;
  }
  .chip:hover { border-color: var(--accent); }
  .row { display: flex; }
  .row.user { justify-content: flex-end; }
  .bubble {
    max-width: min(640px, 88%); padding: 10px 14px; border-radius: 14px;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .user .bubble { background: var(--user-bg); color: var(--user-text); border-bottom-right-radius: 4px; }
  .ai .bubble { background: var(--ai-bg); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
  .meta { margin-top: 8px; font-size: 12px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px 10px; }
  .tag { padding: 1px 7px; border-radius: 6px; background: var(--accent-soft); color: var(--accent); }
  details { margin-top: 6px; font-size: 12px; color: var(--muted); }
  details summary { cursor: pointer; }
  details li { margin: 2px 0; }
  .note { font-size: 12px; color: var(--warn); max-width: 760px; margin: 0 auto; padding: 0 16px; }
  form { border-top: 1px solid var(--border); background: var(--panel); padding: 12px 16px; }
  .inner { max-width: 760px; margin: 0 auto; display: flex; gap: 8px; }
  #q {
    flex: 1; min-width: 0; padding: 11px 14px; border: 1px solid var(--border); border-radius: 12px;
    background: var(--bg); color: var(--text); font: inherit; outline: none;
  }
  #q:focus { border-color: var(--accent); }
  #send {
    padding: 0 18px; border: 0; border-radius: 12px; background: var(--accent); color: #fff;
    font: inherit; font-weight: 600; cursor: pointer;
  }
  #send:disabled { opacity: 0.5; cursor: default; }
  .dots::after { content: "..."; animation: blink 1s steps(4) infinite; }
  @keyframes blink { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; } }
</style>
</head>
<body>
  <header>
    <div class="brand">OwnAI<small>tokenizers, retrieval and GPTs, all built from scratch</small></div>
    <div class="controls">
      <select id="domain" aria-label="Domain"></select>
      <div class="seg" role="group" aria-label="Answer mode">
        <button type="button" data-mode="extractive" class="on">Extractive</button>
        <button type="button" data-mode="generative">Mini-GPT</button>
        <button type="button" data-mode="owngpt" disabled>OwnGPT</button>
      </div>
    </div>
  </header>
  <main id="scroll"><div id="messages"></div></main>
  <div class="note" id="note" hidden></div>
  <form id="chat">
    <div class="inner">
      <input id="q" type="text" autocomplete="off" placeholder="Ask something about the domain..." />
      <button type="submit" id="send">Send</button>
    </div>
  </form>
<script>
  var $ = function (id) { return document.getElementById(id); };
  var messages = $("messages"), scroll = $("scroll"), input = $("q"), send = $("send");
  var select = $("domain"), note = $("note");
  var modeButtons = document.querySelectorAll(".seg button");
  var domains = [], mode = "extractive", hasOwngpt = false, convo = [];
  var NOTES = {
    generative: "Mini-GPT mode: a 4-layer NumPy transformer trained on a tiny corpus. Expect broken text; it shows the model is real, not that it is good.",
    owngpt: "OwnGPT mode: our own GPT, pretrained on French text then fine-tuned to chat. General knowledge, remembers the conversation, can be wrong."
  };

  function current() {
    for (var i = 0; i < domains.length; i++) if (domains[i].id === select.value) return domains[i];
    return null;
  }

  function setMode(m) {
    if (m !== mode) convo = [];  // an OwnGPT conversation does not carry over between modes
    mode = m;
    modeButtons.forEach(function (b) { b.classList.toggle("on", b.dataset.mode === m); });
    note.hidden = !NOTES[m];
    note.textContent = NOTES[m] || "";
    var d = current();
    input.placeholder = m === "owngpt" ? "Écris ton message..."
      : (d && d.language === "fr" ? "Pose une question sur le domaine..." : "Ask something about the domain...");
  }

  function renderEmpty() {
    var d = current();
    convo = [];
    messages.innerHTML = "";
    if (!d) return;
    var box = document.createElement("div");
    box.className = "empty";
    var h = document.createElement("h1");
    h.textContent = d.title;
    var p = document.createElement("div");
    p.textContent = d.chunks + " knowledge chunks indexed" + (d.has_model ? ", mini-GPT available" : "");
    box.appendChild(h); box.appendChild(p);
    var chips = document.createElement("div");
    chips.className = "chips";
    d.examples.forEach(function (q) {
      var c = document.createElement("button");
      c.type = "button"; c.className = "chip"; c.textContent = q;
      c.onclick = function () { ask(q); };
      chips.appendChild(c);
    });
    box.appendChild(chips);
    messages.appendChild(box);
    modeButtons[1].disabled = !d.has_model;
    if (!d.has_model && mode === "generative") setMode("extractive");
    setMode(mode);  // refresh the note and the placeholder for this domain
  }

  function addBubble(role, text) {
    var empty = messages.querySelector(".empty");
    if (empty) empty.remove();
    var row = document.createElement("div");
    row.className = "row " + role;
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    messages.appendChild(row);
    scroll.scrollTop = scroll.scrollHeight;
    return bubble;
  }

  function fillAnswer(bubble, data) {
    bubble.classList.remove("dots");
    bubble.textContent = data.answer || "";
    var meta = document.createElement("div");
    meta.className = "meta";
    var tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = { generative: "mini-GPT", owngpt: "OwnGPT" }[data.mode] || "extractive";
    meta.appendChild(tag);
    if (data.ms != null) {
      var t = document.createElement("span");
      t.textContent = data.ms + " ms";
      meta.appendChild(t);
    }
    bubble.appendChild(meta);
    if (data.sources && data.sources.length) {
      var det = document.createElement("details");
      var sum = document.createElement("summary");
      sum.textContent = data.sources.length + " sources";
      var ul = document.createElement("ul");
      data.sources.forEach(function (s) {
        var li = document.createElement("li"); li.textContent = s; ul.appendChild(li);
      });
      det.appendChild(sum); det.appendChild(ul);
      bubble.appendChild(det);
    }
  }

  function streamChat(pending) {
    // OwnGPT: the whole conversation goes up, the reply streams back token by token.
    var t0 = performance.now(), text = "";
    return fetch("/chat_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: convo })
    }).then(function (r) {
      if (!r.ok || !r.body) throw new Error("stream failed");
      var reader = r.body.getReader(), dec = new TextDecoder();
      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) return;
          text += dec.decode(chunk.value, { stream: true });
          pending.classList.remove("dots");
          pending.textContent = text;
          scroll.scrollTop = scroll.scrollHeight;
          return pump();
        });
      }
      return pump();
    }).then(function () {
      convo.push({ role: "assistant", content: text.trim() });
      fillAnswer(pending, { answer: text.trim(), mode: "owngpt", ms: Math.round(performance.now() - t0) });
    });
  }

  function ask(question) {
    question = (question || "").trim();
    if (!question || send.disabled) return;
    addBubble("user", question);
    input.value = "";
    send.disabled = true;
    var pending = addBubble("ai", "");
    pending.classList.add("dots");
    if (mode === "owngpt") {
      convo.push({ role: "user", content: question });
      streamChat(pending)
        .catch(function () { convo.pop(); pending.classList.remove("dots"); pending.textContent = "(error contacting the server)"; })
        .finally(function () { send.disabled = false; input.focus(); });
      return;
    }
    fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, domain: select.value, mode: mode })
    })
      .then(function (r) { return r.json(); })
      .then(function (data) { fillAnswer(pending, data); })
      .catch(function () { pending.classList.remove("dots"); pending.textContent = "(error contacting the server)"; })
      .finally(function () { send.disabled = false; input.focus(); scroll.scrollTop = scroll.scrollHeight; });
  }

  $("chat").addEventListener("submit", function (e) { e.preventDefault(); ask(input.value); });
  modeButtons.forEach(function (b) { b.onclick = function () { if (!b.disabled) setMode(b.dataset.mode); }; });
  select.onchange = renderEmpty;

  fetch("/domains").then(function (r) { return r.json(); }).then(function (data) {
    domains = data.domains;
    hasOwngpt = !!data.owngpt;
    modeButtons[2].disabled = !hasOwngpt;
    domains.forEach(function (d) {
      var o = document.createElement("option"); o.value = d.id; o.textContent = d.title; select.appendChild(o);
    });
    if (data.default) select.value = data.default;
    renderEmpty();
  });
  input.focus();
</script>
</body>
</html>
"""

N_EXAMPLES = 4


def answer_to_json(answerer, question: str, top_k: int = 4) -> dict:
    """Run one question through an answerer and return a JSON-ready dict.

    Uses the existing ``Answer`` dataclass fields (``.text`` and ``.sources``)
    so both the extractive and generative answerers work unchanged.
    """
    ans = answerer.answer(question, top_k=top_k)
    return {"answer": ans.text, "sources": list(ans.sources)}


class DomainRegistry:
    """Every domain that has a built index, with answerers loaded on first use.

    Loading the retriever and the GPT weights costs a few seconds, so nothing is
    touched until someone actually asks a question in that domain and mode.
    A single in-memory answerer can also be wrapped (``from_answerer``), which
    keeps the handler testable without any artifact on disk.
    """

    def __init__(self):
        self.domains: dict[str, dict] = {}
        self._cache: dict[tuple[str, str], object] = {}
        self._lock = threading.Lock()
        self.owngpt_path: Path | None = None  # artifacts/owngpt/model.pt, trained on Colab

    @classmethod
    def from_answerer(cls, answerer, top_k: int = 4, name: str = "default"):
        reg = cls()
        reg.domains[name] = {"id": name, "title": name, "language": "en", "chunks": 0,
                             "has_model": False, "examples": [], "top_k": top_k}
        reg._cache[(name, "extractive")] = answerer
        return reg

    @classmethod
    def discover(cls, domains_dir="domains", artifacts_dir="artifacts"):
        from ownai.pipeline import load_domain_config

        reg = cls()
        for path in sorted(Path(domains_dir).glob("*.yaml")):
            cfg = load_domain_config(path)
            name = cfg.get("name") or path.stem
            art = Path(artifacts_dir) / name
            if not (art / "index").exists():
                continue
            reg.domains[name] = {
                "id": name,
                "title": cfg.get("display_name") or name,
                "language": cfg.get("language", "en"),
                "chunks": _count_lines(art / "corpus.jsonl"),
                "has_model": (art / "model.npz").exists() and (art / "tokenizer.json").exists(),
                "examples": _examples(cfg),
                "top_k": cfg.get("retrieval", {}).get("top_k", 4),
                "_art": art,
            }
        owngpt = Path(artifacts_dir) / "owngpt" / "model.pt"
        reg.owngpt_path = owngpt if owngpt.exists() else None
        return reg

    def public(self) -> list[dict]:
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in self.domains.values()]

    def answerer(self, name: str, mode: str):
        key = ("*", mode) if mode == "owngpt" else (name, mode)  # OwnGPT is shared by all domains
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self._load(name, mode)
            return self._cache[key]

    def _load(self, name: str, mode: str):
        from ownai.rag import ExtractiveAnswerer, GenerativeAnswerer

        if mode == "owngpt":
            return OwnGPTAnswerer(self.owngpt_path)

        art = self.domains[name]["_art"]
        other = (name, "generative" if mode == "extractive" else "extractive")
        retr = self._cache[other].retriever if other in self._cache else None
        if retr is None:
            from ownai.retrieval import HybridRetriever

            retr = HybridRetriever.load(art / "index")
        if mode == "generative":
            from ownai.model import GPT
            from ownai.tokenizer import BPETokenizer

            return GenerativeAnswerer(retr, GPT.load(art / "model.npz"), BPETokenizer.load(art / "tokenizer.json"))
        return ExtractiveAnswerer(retr)

    def ask(self, question: str, name: str | None = None, mode: str = "extractive") -> dict:
        if name not in self.domains:
            name = next(iter(self.domains))
        dom = self.domains[name]
        available = {"generative": dom["has_model"], "owngpt": self.owngpt_path is not None}
        if not available.get(mode):
            mode = "extractive"
        if not question:
            return {"answer": "Ask me a question.", "sources": [], "mode": mode, "domain": name, "ms": 0}
        t0 = time.perf_counter()
        result = answer_to_json(self.answerer(name, mode), question, top_k=dom["top_k"])
        result.update(mode=mode, domain=name, ms=round((time.perf_counter() - t0) * 1000))
        return result


class OwnGPTAnswerer:
    """Adapter so the PyTorch OwnGPT answers through the same ``.answer`` API.

    One generation at a time: the model is shared by every request thread.
    """

    def __init__(self, path):
        from ownai.gpt.chat import OwnGPT

        self.bot = OwnGPT.load(path)
        self.lock = threading.Lock()

    def answer(self, question: str, top_k: int = 4):
        from ownai.rag.answer import Answer

        with self.lock:
            return Answer(text=self.bot.reply([{"role": "user", "content": question}]))

    def stream(self, messages):
        with self.lock:
            yield from self.bot.stream_reply(messages)


MAX_TURNS, MAX_CHARS = 20, 4000


def clean_messages(raw) -> list[dict]:
    """Keep the last user/assistant turns of a client-sent conversation, bounded in size."""
    if not isinstance(raw, list):
        return []
    msgs = [{"role": m["role"], "content": m["content"][:MAX_CHARS]} for m in raw
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    return msgs[-MAX_TURNS:]


def _count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _examples(cfg) -> list[str]:
    qa_path = cfg.get("eval", {}).get("qa_path")
    try:
        qa = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    return [q["question"] for q in qa if q.get("in_domain", True)][:N_EXAMPLES]


def build_handler(source, top_k: int = 4, default: str | None = None):
    """Return a BaseHTTPRequestHandler subclass closed over a registry.

    ``source`` is a ``DomainRegistry`` or a bare answerer (wrapped on the fly).
    Factored out of ``serve`` so tests can assert on the class without ever
    starting a server: GET / serves the page, GET /domains lists domains,
    POST /ask answers a question.
    """
    registry = source if isinstance(source, DomainRegistry) else DomainRegistry.from_answerer(source, top_k)
    default = default if default in registry.domains else next(iter(registry.domains), None)

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

        def _json(self, code, obj):
            self._send(code, "application/json; charset=utf-8", json.dumps(obj).encode("utf-8"))

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE_HTML.encode("utf-8"))
            elif path == "/domains":
                self._json(200, {"domains": registry.public(), "default": default,
                                 "owngpt": registry.owngpt_path is not None})
            else:
                self._send(404, "text/plain; charset=utf-8", b"Not Found")

        def _chat_stream(self, payload):
            messages = clean_messages(payload.get("messages"))
            if registry.owngpt_path is None or not messages or messages[-1]["role"] != "user":
                self._send(400, "text/plain; charset=utf-8", b"OwnGPT unavailable or empty conversation")
                return
            answerer = registry.answerer("*", "owngpt")
            # No Content-Length: the body is written as it is generated, then the
            # connection closes (HTTP/1.0), which tells the browser the reply is done.
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for piece in answerer.stream(messages):
                self.wfile.write(piece.encode("utf-8"))
                self.wfile.flush()

        def do_POST(self):
            if self.path.split("?")[0] == "/chat_stream":
                length = int(self.headers.get("Content-Length", 0) or 0)
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                except ValueError:
                    payload = {}
                self._chat_stream(payload if isinstance(payload, dict) else {})
                return
            if self.path.split("?")[0] != "/ask":
                self._send(404, "text/plain; charset=utf-8", b"Not Found")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                question = (payload.get("question") or "").strip()
            except (ValueError, AttributeError):
                self._json(400, {"answer": "Invalid request.", "sources": []})
                return
            try:
                result = registry.ask(question, payload.get("domain") or default, payload.get("mode", "extractive"))
            except Exception as exc:  # keep the server alive, surface the error in the chat
                self._json(500, {"answer": f"Server error: {exc}", "sources": []})
                return
            self._json(200, result)

    return ChatHandler


def _artifacts_name(cfg, domain_path) -> str:
    return cfg.get("name") or Path(domain_path).stem


def serve(domain_path=None, host: str = "127.0.0.1", port: int = 8000, generative: bool = False):
    """Serve the chat UI over every built domain until interrupted.

    ``domain_path`` only picks the domain selected on load. ``generative`` is
    kept for CLI compatibility: the mode is now switched live in the page.
    """
    from ownai.pipeline import load_domain_config

    registry = DomainRegistry.discover()
    if not registry.domains:
        raise SystemExit("No built domain found. Run: python -m ownai.cli ingest/index --domain domains/<x>.yaml")
    default = _artifacts_name(load_domain_config(domain_path), domain_path) if domain_path else None
    httpd = ThreadingHTTPServer((host, port), build_handler(registry, default=default))
    url = f"http://{host}:{port}"
    print(f"OwnAI web chat serving at {url}  ({len(registry.domains)} domains, Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        httpd.server_close()
