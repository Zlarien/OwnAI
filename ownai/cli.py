"""Command-line interface for the whole OwnAI workflow.

    python -m ownai.cli ingest    --domain domains/mario-wii.yaml
    python -m ownai.cli index     --domain domains/mario-wii.yaml
    python -m ownai.cli tokenizer --domain domains/mario-wii.yaml
    python -m ownai.cli train     --domain domains/mario-wii.yaml --steps 2000
    python -m ownai.cli chat      --domain domains/mario-wii.yaml
    python -m ownai.cli eval      --domain domains/mario-wii.yaml

Every domain is described by a small YAML file, so the exact same code
specializes to any subject just by swapping the config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ownai.pipeline import (
    build_corpus_from_paths,
    build_corpus_from_wiki,
    build_index_from_corpus,
    corpus_to_token_stream,
    load_domain_config,
)


def _artifacts_dir(cfg, domain_path) -> Path:
    name = cfg.get("name") or Path(domain_path).stem
    d = Path("artifacts") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _corpus_path(cfg, domain_path) -> Path:
    return _artifacts_dir(cfg, domain_path) / "corpus.jsonl"


# --------------------------------------------------------------------- ingest
def cmd_ingest(args):
    cfg = load_domain_config(args.domain)
    corpus_path = _corpus_path(cfg, args.domain)
    src = cfg.get("source", {})
    chunk = cfg.get("chunking", {})
    if src.get("type") == "wiki":
        print(f"Downloading wiki category {src['category']!r} from {src['api_url']} ...")
        chunks = build_corpus_from_wiki(
            src["api_url"], src["category"], corpus_path,
            limit=src.get("limit", 500),
            max_words=chunk.get("max_words", 180), overlap=chunk.get("overlap", 30),
        )
    else:  # local files
        paths = [Path(p) for p in src.get("paths", [])]
        print(f"Ingesting local files from: {', '.join(map(str, paths))}")
        chunks = build_corpus_from_paths(
            paths, corpus_path,
            max_words=chunk.get("max_words", 180), overlap=chunk.get("overlap", 30),
        )
    print(f"Wrote {len(chunks)} chunks -> {corpus_path}")


# ---------------------------------------------------------------------- index
def cmd_index(args):
    cfg = load_domain_config(args.domain)
    corpus_path = _corpus_path(cfg, args.domain)
    r = cfg.get("retrieval", {})
    print("Building hybrid retriever (BM25 + word2vec)...")
    retr = build_index_from_corpus(
        corpus_path,
        embed_dim=r.get("embed_dim", 64),
        w2v_epochs=r.get("w2v_epochs", 10),
        alpha=r.get("alpha", 0.5),
        seed=cfg.get("seed", 0),
    )
    out = _artifacts_dir(cfg, args.domain) / "index"
    retr.save(out)
    print(f"Saved index -> {out}  ({len(retr.chunks)} chunks)")


# ------------------------------------------------------------------ tokenizer
def cmd_tokenizer(args):
    from ownai.data import read_corpus
    from ownai.tokenizer import BPETokenizer

    cfg = load_domain_config(args.domain)
    corpus_path = _corpus_path(cfg, args.domain)
    m = cfg.get("model", {})
    chunks = read_corpus(corpus_path)
    tok = BPETokenizer(special_tokens=["<|bos|>", "<|sep|>", "<|eos|>"])
    print(f"Training BPE tokenizer (vocab_size={m.get('vocab_size', 4000)})...")
    tok.train([c.text for c in chunks], vocab_size=m.get("vocab_size", 4000), verbose=True)
    out = _artifacts_dir(cfg, args.domain) / "tokenizer.json"
    tok.save(out)
    print(f"Saved tokenizer -> {out}  (vocab {tok.vocab_size})")


# ---------------------------------------------------------------------- train
def cmd_train(args):
    from ownai.model import GPT, GPTConfig
    from ownai.model.train import train_lm
    from ownai.tokenizer import BPETokenizer

    cfg = load_domain_config(args.domain)
    art = _artifacts_dir(cfg, args.domain)
    m = cfg.get("model", {})
    tok = BPETokenizer.load(art / "tokenizer.json")
    tokens = corpus_to_token_stream(_corpus_path(cfg, args.domain), tok)
    print(f"Token stream length: {len(tokens)}")

    gpt_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=m.get("block_size", 128),
        n_layer=m.get("n_layer", 4),
        n_head=m.get("n_head", 4),
        n_embd=m.get("n_embd", 128),
        dropout=m.get("dropout", 0.1),
    )
    model = GPT(gpt_cfg)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    history = train_lm(
        model, tokens,
        steps=args.steps, batch_size=m.get("batch_size", 16),
        lr=m.get("lr", 3e-4), warmup=m.get("warmup", 100),
        weight_decay=m.get("weight_decay", 0.01), seed=cfg.get("seed", 0),
        log_every=args.log_every,
    )
    model.save(art / "model.npz")
    (art / "train_history.json").write_text(json.dumps(history), encoding="utf-8")
    print(f"Saved model -> {art / 'model.npz'}")
    _maybe_plot(history, art / "loss_curve.png")


def _maybe_plot(history, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(history["step"], history["loss"])
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        print(f"Saved loss curve -> {path}")
    except Exception as e:  # matplotlib optional
        print(f"(skipped loss plot: {e})")


# ----------------------------------------------------------------------- chat
def cmd_chat(args):
    from ownai.rag import ExtractiveAnswerer, GenerativeAnswerer
    from ownai.retrieval import HybridRetriever

    cfg = load_domain_config(args.domain)
    art = _artifacts_dir(cfg, args.domain)
    retr = HybridRetriever.load(art / "index")

    if args.generative:
        from ownai.model import GPT
        from ownai.tokenizer import BPETokenizer

        model = GPT.load(art / "model.npz")
        tok = BPETokenizer.load(art / "tokenizer.json")
        answerer = GenerativeAnswerer(retr, model, tok)
        mode = "generative (from-scratch mini-GPT)"
    else:
        answerer = ExtractiveAnswerer(retr)
        mode = "extractive (faithful to the corpus)"

    title = cfg.get("display_name", cfg.get("name", "OwnAI"))
    print(f"\n=== {title} — mode: {mode} ===")
    print("Ask a question (empty line or 'quit' to exit).\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        ans = answerer.answer(q, top_k=cfg.get("retrieval", {}).get("top_k", 4))
        print(f"\nai  > {ans.text}")
        if ans.sources:
            print("      sources: " + "; ".join(ans.sources))
        print()


# ----------------------------------------------------------------------- eval
def cmd_eval(args):
    from ownai.eval import evaluate_retrieval
    from ownai.retrieval import HybridRetriever

    cfg = load_domain_config(args.domain)
    art = _artifacts_dir(cfg, args.domain)
    retr = HybridRetriever.load(art / "index")

    qa_path = Path(cfg["eval"]["qa_path"])
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    report = evaluate_retrieval(retr, qa, ks=tuple(cfg["eval"].get("ks", [1, 3, 5])))

    lines = ["# Retrieval evaluation\n", f"Queries: {report['n_queries']}\n", "\n| metric | value |", "| --- | --- |"]
    for key, val in report.items():
        if key == "n_queries":
            continue
        lines.append(f"| {key} | {val:.3f} |")
    md = "\n".join(lines) + "\n"
    out = art / "eval_report.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved report -> {out}")


def cmd_serve(args):
    from ownai.web import serve

    serve(args.domain, host=args.host, port=args.port, generative=args.generative)


def cmd_eval_answers(args):
    from ownai.eval import evaluate_answers
    from ownai.rag import ExtractiveAnswerer
    from ownai.retrieval import HybridRetriever

    cfg = load_domain_config(args.domain)
    art = _artifacts_dir(cfg, args.domain)
    retr = HybridRetriever.load(art / "index")
    answerer = ExtractiveAnswerer(retr)
    qa = json.loads(Path(cfg["eval"]["qa_path"]).read_text(encoding="utf-8"))
    report = evaluate_answers(answerer, qa)
    for key, val in report.items():
        print(f"  {key}: {val}")


def cmd_eval_lm(args):
    from ownai.model import GPT
    from ownai.model.train import evaluate_perplexity
    from ownai.tokenizer import BPETokenizer

    cfg = load_domain_config(args.domain)
    art = _artifacts_dir(cfg, args.domain)
    model = GPT.load(art / "model.npz")
    tok = BPETokenizer.load(art / "tokenizer.json")
    tokens = corpus_to_token_stream(_corpus_path(cfg, args.domain), tok)
    ppl = evaluate_perplexity(
        model, tokens, block_size=model.cfg.block_size, batch_size=8, seed=cfg.get("seed", 0), batches=args.batches
    )
    print(f"Mini-GPT perplexity on the corpus: {ppl:.2f}  (lower is better)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ownai", description="Domain AI, 100% from scratch.")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--domain", required=True, help="path to a domain YAML config")
        sp.set_defaults(func=fn)
        return sp

    add("ingest", cmd_ingest, "download/read sources into a chunked corpus")
    add("index", cmd_index, "build and save the hybrid retriever")
    add("tokenizer", cmd_tokenizer, "train the BPE tokenizer on the corpus")
    tr = add("train", cmd_train, "train the mini-GPT language model")
    tr.add_argument("--steps", type=int, default=2000)
    tr.add_argument("--log-every", type=int, default=50)
    ch = add("chat", cmd_chat, "chat with the domain AI in the terminal")
    ch.add_argument("--generative", action="store_true", help="use the mini-GPT instead of extractive answers")
    add("eval", cmd_eval, "evaluate retrieval quality (Recall@k, MRR)")
    add("eval-answers", cmd_eval_answers, "evaluate answer quality (keyword hit-rate, abstention)")
    el = add("eval-lm", cmd_eval_lm, "evaluate the mini-GPT language model (perplexity)")
    el.add_argument("--batches", type=int, default=20)
    sv = add("serve", cmd_serve, "serve the web chat UI (browser)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--generative", action="store_true", help="use the mini-GPT instead of extractive answers")
    return p


def main(argv=None):
    # Ensure non-ASCII I/O works on any console (e.g. Windows cp1252): accented
    # questions must be read and answers printed correctly.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
