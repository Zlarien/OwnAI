# OwnAI — Roadmap

A concrete, non-hype plan for where OwnAI goes next. The current release is a
complete, tested, from-scratch stack (autograd → NN → BPE → BM25 + word2vec →
mini-GPT → RAG → eval), with the **extractive mode as the reliable default** and
an honest confidence gate. The items below sharpen the weak spots called out in
[`ARCHITECTURE.md` §9](ARCHITECTURE.md) rather than bolting on features for their
own sake.

Each item lists the *why* and a rough sense of the work — no dates, no promises,
just the credible next steps.

---

## Near-term

Things that are natural extensions of what already works, mostly data- and
polish-focused.

- **Full-wiki corpus + Colab retrain.** The `type: wiki` ingestion path already
  works against the live MediaWiki export API (verified on the real Super Mario
  Wiki). The next step is to build a genuinely large corpus from a full category
  tree and retrain the mini-GPT on a free Colab GPU (NumPy → CuPy transparently
  via `backend.py`). This is the single highest-leverage change: it directly
  attacks the small-corpus overfitting that keeps generative output incoherent
  today. A `domains/mario-wii-full.yaml` config already sketches this direction.
- **Web UI polish.** The stdlib `http.server` chat UI (`ownai/web/`) needs
  hardening: streaming/typed answer rendering, visible source citations with
  links back to the corpus, a language/domain switcher, and graceful handling of
  the "I don't know" path so refusals read as a feature, not a failure. Still
  zero external web dependencies.
- **Top-p (nucleus) generation.** `GPT.generate` currently supports temperature
  and top-k. Adding top-p sampling gives better quality/diversity trade-offs for
  the generative mode on larger corpora, and it is a small, self-contained,
  testable addition to the sampling loop.

## Mid-term

Deeper improvements to model and evaluation quality.

- **Better tokenizer coverage.** Grow BPE vocab and training data so common
  domain terms become single tokens, and add a light pre-tokenization/regex
  split (GPT-2 style) so merges don't cross word/punctuation boundaries awkwardly.
  Better token economy means more effective context per `block_size`.
- **Larger model.** Scale `n_layer`/`n_embd`/`block_size` now that a real corpus
  justifies the capacity, with attention to training stability (the cosine
  schedule and weight decay are already in place). Report parameter count and
  perplexity as the corpus grows.
- **Answer-level evaluation dashboards.** Retrieval is already measured with
  Recall@k and MRR. The next layer is *answer* quality: an offline harness that
  scores extractive answers against a gold QA set (exact/─overlap and
  source-correctness), tracks the confidence gate's precision/recall (does it
  refuse when it should, and only then?), and renders a small dashboard so
  regressions are visible at a glance.

## Stretch

Larger structural bets, worth doing once the fundamentals above are solid.

- **Multi-domain routing.** Today one index answers one domain. A lightweight
  router (reusing the existing IDF/informative-term machinery) could pick the
  right domain index per question — or honestly refuse across all of them —
  turning OwnAI into a small federation of specialists without giving up the
  refuse-when-unsure guarantee.
- **French (and other-language) corpora at scale.** The engine is already proven
  subject- and language-agnostic (the shipped `systeme-solaire` French domain).
  The stretch goal is building substantial French corpora from French MediaWiki
  sources and validating that retrieval, the bilingual stopword gate, and
  generation all hold up at scale in a second language — demonstrating true
  language-agnosticism, not just a token demo.

---

*Principle for all of the above: every addition stays true to the project's
core — no external AI frameworks, NumPy (or CuPy) only, and always measurable.*
