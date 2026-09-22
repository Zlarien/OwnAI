"""Data for OwnGPT: corpora recipes, tokenized binary shards, batch loaders.

Two stages, two formats:

  * pretraining : a flat stream of token ids (uint16) in ``train_000.bin``,
    ``train_001.bin``... plus ``val_000.bin``. Every document starts with the
    end-of-text token. Batches are random windows, so resuming never needs to
    remember a read position.
  * chat (SFT)  : conversations rendered with the chat template below. For each
    split we store ``{split}_ids.bin`` (uint16), ``{split}_mask.bin`` (uint8,
    1 = assistant token the model must learn) and ``{split}_offsets.npy``.

Every data directory carries a copy of ``tokenizer.json`` so a run always knows
which vocabulary its ids belong to.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ownai.gpt.tokenizer import Tokenizer

# Where each language's text comes from. Weights mix pretraining sources;
# ``exclude`` drops rows whose field contains a substring.
RECIPES = {
    "fr": {
        "pretrain": [
            # FineWeb2 French, filtered to its top ~10% by an educational-quality classifier.
            {"dataset": "epfml/FineWeb2-HQ", "subset": "fra_Latn", "weight": 0.7, "columns": ["text"]},
            {"dataset": "wikimedia/wikipedia", "subset": "20231101.fr", "weight": 0.3, "columns": ["text"]},
        ],
        "sft": [
            {"dataset": "jpacifico/French-Alpaca-dataset-Instruct-110K"},
            # Translated chat corpora; the translated logic-puzzle sets are unreadable, so they go.
            {"dataset": "angeluriot/french_instruct", "exclude": {"source": "Logic"}},
        ],
    },
    "en": {
        "pretrain": [{"dataset": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "weight": 1.0,
                      "columns": ["text"]}],
        "sft": [{"dataset": "HuggingFaceTB/smol-smoltalk"}],
    },
}

IDENTITY = {"fr": Path(__file__).with_name("identity_fr.jsonl")}
EMPTY_INPUTS = {"", "aucun", "aucune", "aucune entrée", "n/a", "none", "nan", "-"}


# --------------------------------------------------------------------------- chat template

def render_conversation(messages, tok: Tokenizer):
    """Token ids + learn-mask for a list of {"role", "content"} messages.

    <|user|> text <|assistant|> reply <|endoftext|> ... Only assistant replies and
    the end-of-text that closes them are learned; the rest is context.
    """
    ids, mask = [], []
    for msg in messages:
        content = (msg.get("content") or "").strip()
        if msg.get("role") == "assistant":
            body = tok.encode(content) + [tok.eot]
            ids += [tok.assistant] + body
            mask += [0] + [1] * len(body)
        else:  # user or system
            part = [tok.user] + tok.encode(content)
            ids += part
            mask += [0] * len(part)
    return ids, mask


def render_prompt(messages, tok: Tokenizer):
    """Ids of a conversation ready for the model to write the next assistant reply."""
    ids, _ = render_conversation(messages, tok)
    return ids + [tok.assistant]


def to_messages(row: dict):
    """Normalize the common chat dataset layouts to [{"role", "content"}]."""
    if "messages" in row:
        return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    if "conversation" in row:
        msgs = [{"role": m["role"], "content": m.get("text") or m.get("content", "")} for m in row["conversation"]]
        if row.get("context"):
            msgs.insert(0, {"role": "system", "content": row["context"]})
        return msgs
    if "conversations" in row:
        roles = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
        return [{"role": roles.get(m["from"], "user"), "content": m["value"]} for m in row["conversations"]]
    if "instruction" in row and "output" in row:
        extra = (row.get("input") or "").strip()
        if extra.lower().strip(".") in EMPTY_INPUTS:  # French-Alpaca writes "Aucun" for no input
            extra = ""
        prompt = row["instruction"] + (f"\n\n{extra}" if extra else "")
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["output"]}]
    return None


# --------------------------------------------------------------------------- text sources

def _stream(src: dict):
    from datasets import load_dataset

    kw = {"columns": src["columns"]} if src.get("columns") else {}
    return load_dataset(src["dataset"], name=src.get("subset"), split=src.get("split", "train"),
                        streaming=True, **kw)


def iter_texts(sources, seed: int = 0, text_files=None):
    """Documents from several sources, interleaved at random by their weights."""
    if text_files:
        for f in text_files:
            yield Path(f).read_text(encoding="utf-8")
        return
    rng = np.random.default_rng(seed)
    live = [(iter(_stream(s)), s.get("weight", 1.0), s.get("text_field", "text")) for s in sources]
    while live:
        w = np.array([x[1] for x in live])
        i = rng.choice(len(live), p=w / w.sum())
        try:
            yield next(live[i][0])[live[i][2]]
        except StopIteration:
            live.pop(i)


def train_tokenizer(out, recipe="fr", vocab_size=32768, max_chars=300_000_000, text_files=None):
    from ownai.gpt.tokenizer import train_bpe

    texts = iter_texts(RECIPES[recipe]["pretrain"], seed=1, text_files=text_files)
    tok = train_bpe(texts, vocab_size=vocab_size, max_chars=max_chars, lang=recipe)
    tok.save(out)
    print(f"saved {out}: {tok.n_vocab:,} tokens", flush=True)
    return tok


# --------------------------------------------------------------------------- pretraining shards

class _ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int):
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, shard_tokens
        self.buf, self.n, self.index, self.total = [], 0, 0, 0

    def add(self, arr: np.ndarray):
        self.buf.append(arr)
        self.n += len(arr)
        self.total += len(arr)
        if self.n >= self.shard_tokens:
            self.flush()

    def flush(self):
        if not self.n:
            return
        path = self.out_dir / f"{self.prefix}_{self.index:03d}.bin"
        np.concatenate(self.buf).astype(np.uint16).tofile(path)
        print(f"  wrote {path.name}: {self.n:,} tokens", flush=True)
        self.buf, self.n = [], 0
        self.index += 1


def _attach_tokenizer(out: Path, tokenizer_path) -> Tokenizer:
    out.mkdir(parents=True, exist_ok=True)
    if Path(tokenizer_path).resolve() != (out / "tokenizer.json").resolve():
        shutil.copy(tokenizer_path, out / "tokenizer.json")
    tok = Tokenizer.load(out / "tokenizer.json")
    assert tok.n_vocab <= 65536, "uint16 shards need a vocabulary under 65536"
    return tok


def prepare_pretrain(out_dir, tokenizer_path, total_tokens: int, recipe="fr", val_tokens: int = 2_000_000,
                     shard_tokens: int = 100_000_000, text_files=None, batch_docs: int = 1024):
    """Stream the recipe's corpora, tokenize them and write uint16 shards until ``total_tokens``."""
    out = Path(out_dir)
    tok = _attach_tokenizer(out, tokenizer_path)
    val = _ShardWriter(out, "val", val_tokens)
    train = _ShardWriter(out, "train", shard_tokens)

    def consume(batch):
        for ids in tok.encode_batch(batch):
            arr = np.array([tok.eot] + ids, dtype=np.uint16)
            (val if val.total < val_tokens else train).add(arr)

    batch, n_chars = [], 0
    for text in iter_texts(RECIPES[recipe]["pretrain"], text_files=text_files):
        batch.append(text)
        n_chars += len(text)
        if len(batch) >= batch_docs:
            consume(batch)
            batch = []
            print(f"  {train.total:,} / {total_tokens:,} train tokens", flush=True)
            if train.total >= total_tokens:
                break
    if batch:
        consume(batch)
    val.flush()
    train.flush()
    meta = {"stage": "pretrain", "recipe": recipe, "train_tokens": train.total, "val_tokens": val.total,
            "chars_per_token": round(n_chars / max(1, train.total + val.total), 2),
            "source": "text files" if text_files else [s["dataset"] for s in RECIPES[recipe]["pretrain"]]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done: {train.total:,} train tokens, {val.total:,} val tokens", flush=True)
    return meta


# --------------------------------------------------------------------------- chat (SFT) data

def iter_conversations(sources, max_per_source=None, jsonl=None):
    if jsonl:
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield to_messages(json.loads(line))
        return
    for src in sources:
        n = 0
        for row in _stream(src):
            if any(bad in str(row.get(field, "")) for field, bad in src.get("exclude", {}).items()):
                continue
            msgs = to_messages(row)
            if msgs:
                yield msgs
                n += 1
                if max_per_source and n >= max_per_source:
                    break
        print(f"  {src['dataset']}: {n:,} conversations", flush=True)


def prepare_sft(out_dir, tokenizer_path, recipe="fr", max_per_source: int | None = None, val_every: int = 100,
                jsonl=None, identity_repeat: int = 50):
    """Render chat conversations into ids/mask/offsets files (1 in ``val_every`` goes to val).

    The identity conversations (who OwnGPT is) are mixed in ``identity_repeat``
    times so the model reliably knows its own name and maker.
    """
    out = Path(out_dir)
    tok = _attach_tokenizer(out, tokenizer_path)
    parts = {"sft_train": ([], [], [0]), "sft_val": ([], [], [0])}

    def add(messages, split):
        ids, mask = render_conversation(messages, tok)
        if not any(mask):
            return
        all_ids, all_mask, offsets = parts[split]
        all_ids.append(np.array(ids, dtype=np.uint16))
        all_mask.append(np.array(mask, dtype=np.uint8))
        offsets.append(offsets[-1] + len(ids))

    n = 0
    for n, messages in enumerate(iter_conversations(RECIPES[recipe]["sft"], max_per_source, jsonl), start=1):
        add(messages, "sft_val" if n % val_every == 0 else "sft_train")
    identity = IDENTITY.get(recipe)
    if identity and identity.exists() and not jsonl:
        convs = [to_messages(json.loads(line)) for line in identity.read_text(encoding="utf-8").splitlines() if line]
        for _ in range(identity_repeat):
            for messages in convs:
                add(messages, "sft_train")
    for name, (all_ids, all_mask, offsets) in parts.items():
        if all_ids:
            np.concatenate(all_ids).tofile(out / f"{name}_ids.bin")
            np.concatenate(all_mask).tofile(out / f"{name}_mask.bin")
            np.save(out / f"{name}_offsets.npy", np.array(offsets, dtype=np.int64))
    meta = {"stage": "sft", "recipe": recipe, "conversations": n,
            "source": str(jsonl) if jsonl else [s["dataset"] for s in RECIPES[recipe]["sft"]]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done: {n:,} conversations", flush=True)
    return meta


# --------------------------------------------------------------------------- batch loaders

def _to_device(x, y, device):
    import torch

    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if str(device).startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


class PretrainLoader:
    """Random (B, T) windows across all shards of a split."""

    def __init__(self, data_dir, split: str, batch_size: int, block_size: int, device="cpu", seed: int = 0):
        files = sorted(Path(data_dir).glob(f"{split}_*.bin"))
        if not files:
            raise FileNotFoundError(f"no {split}_*.bin shard in {data_dir}")
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.shards = [s for s in self.shards if len(s) > block_size + 1]
        sizes = np.array([len(s) for s in self.shards], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.B, self.T, self.device = batch_size, block_size, device
        self.rng = np.random.default_rng(seed)

    def next_batch(self):
        x = np.empty((self.B, self.T), dtype=np.int64)
        y = np.empty((self.B, self.T), dtype=np.int64)
        for i in range(self.B):
            shard = self.shards[self.rng.choice(len(self.shards), p=self.weights)]
            start = self.rng.integers(0, len(shard) - self.T - 1)
            window = shard[start:start + self.T + 1].astype(np.int64)
            x[i], y[i] = window[:-1], window[1:]
        return _to_device(x, y, self.device)


class SFTLoader:
    """Rows packed with whole conversations; non-assistant targets are -1 (ignored)."""

    def __init__(self, data_dir, split: str, batch_size: int, block_size: int, device="cpu", seed: int = 0):
        d = Path(data_dir)
        name = f"sft_{split}"
        self.ids = np.memmap(d / f"{name}_ids.bin", dtype=np.uint16, mode="r")
        self.mask = np.memmap(d / f"{name}_mask.bin", dtype=np.uint8, mode="r")
        self.offsets = np.load(d / f"{name}_offsets.npy")
        self.B, self.T, self.device = batch_size, block_size, device
        self.rng = np.random.default_rng(seed)

    def next_batch(self):
        n_conv = len(self.offsets) - 1
        x = np.empty((self.B, self.T), dtype=np.int64)
        y = np.empty((self.B, self.T), dtype=np.int64)
        for i in range(self.B):
            ids, mask = [], []
            while len(ids) < self.T + 1:
                c = self.rng.integers(0, n_conv)
                a, b = self.offsets[c], self.offsets[c + 1]
                ids.extend(self.ids[a:b].tolist())
                mask.extend(self.mask[a:b].tolist())
            ids = np.array(ids[: self.T + 1], dtype=np.int64)
            mask = np.array(mask[: self.T + 1], dtype=bool)
            x[i] = ids[:-1]
            y[i] = np.where(mask[1:], ids[1:], -1)
        return _to_device(x, y, self.device)
