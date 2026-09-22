"""OwnGPT command line. The full French pipeline, in order:

    python -m ownai.gpt train-tokenizer
    python -m ownai.gpt prepare-pretrain --tokens 1e9
    python -m ownai.gpt train --stage pretrain --preset t4 --hours 11
    python -m ownai.gpt prepare-sft
    python -m ownai.gpt train --stage sft --init runs/pretrain/model.pt
    python -m ownai.gpt chat runs/sft/model.pt

Every training command resumes by itself when relaunched.
"""
from __future__ import annotations

import argparse
import sys

TOKENIZER = "data/owngpt/tokenizer.json"


def main(argv=None):
    for stream in (sys.stdin, sys.stdout, sys.stderr):  # samples may hold any unicode (Windows cp1252)
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="python -m ownai.gpt")
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("train-tokenizer", help="learn our own BPE vocabulary")
    pk.add_argument("--out", default=TOKENIZER)
    pk.add_argument("--recipe", default="fr", choices=["fr", "en"])
    pk.add_argument("--vocab-size", type=int, default=32768)
    pk.add_argument("--chars", type=float, default=2e8, help="characters of text to learn from")
    pk.add_argument("--text-files", nargs="*", help="learn from local files instead of the recipe")

    pp = sub.add_parser("prepare-pretrain", help="tokenize the recipe's corpora into shards")
    pp.add_argument("--out", default="data/owngpt/pretrain")
    pp.add_argument("--tokenizer", default=TOKENIZER)
    pp.add_argument("--recipe", default="fr", choices=["fr", "en"])
    pp.add_argument("--tokens", type=float, default=1e9, help="train tokens to write (e.g. 1e9)")
    pp.add_argument("--val-tokens", type=float, default=2e6)
    pp.add_argument("--shard-tokens", type=float, default=1e8)
    pp.add_argument("--text-files", nargs="*", help="use local files instead of the recipe")

    ps = sub.add_parser("prepare-sft", help="render chat conversations for fine-tuning")
    ps.add_argument("--out", default="data/owngpt/sft")
    ps.add_argument("--tokenizer", default=TOKENIZER)
    ps.add_argument("--recipe", default="fr", choices=["fr", "en"])
    ps.add_argument("--max-per-source", type=int)
    ps.add_argument("--jsonl", help="local jsonl of conversations instead of the recipe")

    pt = sub.add_parser("train", help="pretrain or fine-tune (resumes automatically)")
    pt.add_argument("--stage", choices=["pretrain", "sft"], default="pretrain")
    pt.add_argument("--data", default=None)
    pt.add_argument("--out", default=None)
    pt.add_argument("--preset", default="t4")
    pt.add_argument("--init", help="checkpoint to start from (sft)")
    pt.add_argument("--max-steps", type=int, default=5000)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--grad-accum", type=int, default=4)
    pt.add_argument("--lr", type=float)
    pt.add_argument("--warmup", type=int)
    pt.add_argument("--eval-every", type=int, default=250)
    pt.add_argument("--save-every", type=int, default=500)
    pt.add_argument("--hours", type=float, help="stop and save after this many hours")
    pt.add_argument("--compile", action="store_true")
    pt.add_argument("--device")

    pc = sub.add_parser("chat", help="talk to a trained checkpoint")
    pc.add_argument("checkpoint")
    pc.add_argument("--device")

    a = p.parse_args(argv)
    if a.cmd == "train-tokenizer":
        from ownai.gpt.data import train_tokenizer

        train_tokenizer(a.out, recipe=a.recipe, vocab_size=a.vocab_size, max_chars=int(a.chars),
                        text_files=a.text_files)
    elif a.cmd == "prepare-pretrain":
        from ownai.gpt.data import prepare_pretrain

        prepare_pretrain(a.out, a.tokenizer, int(a.tokens), recipe=a.recipe, val_tokens=int(a.val_tokens),
                         shard_tokens=int(a.shard_tokens), text_files=a.text_files)
    elif a.cmd == "prepare-sft":
        from ownai.gpt.data import prepare_sft

        prepare_sft(a.out, a.tokenizer, recipe=a.recipe, max_per_source=a.max_per_source, jsonl=a.jsonl)
    elif a.cmd == "train":
        from ownai.gpt.train import train

        train(stage=a.stage, data=a.data or f"data/owngpt/{a.stage}", out=a.out or f"runs/{a.stage}",
              preset=a.preset, init=a.init, max_steps=a.max_steps, batch_size=a.batch_size,
              grad_accum=a.grad_accum, lr=a.lr, warmup=a.warmup, eval_every=a.eval_every,
              save_every=a.save_every, hours=a.hours, compile=a.compile, device=a.device)
    elif a.cmd == "chat":
        from ownai.gpt.chat import main as chat_main

        chat_main(a.checkpoint, a.device)


if __name__ == "__main__":
    main()
