"""Tests for OwnGPT (the PyTorch GPT). Skipped when the [gpt] extra is not installed."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tiktoken")
pytest.importorskip("regex")

from ownai.gpt import GPT, config_from_preset  # noqa: E402
from ownai.gpt.data import (  # noqa: E402
    IDENTITY, SFTLoader, prepare_pretrain, prepare_sft, render_conversation, render_prompt, to_messages,
)
from ownai.gpt.tokenizer import Tokenizer, train_bpe  # noqa: E402
from ownai.gpt.train import load_checkpoint, lr_at, tokenizer_of, train  # noqa: E402

CONV = [{"role": "user", "content": "Quelle est la capitale de la France ?"},
        {"role": "assistant", "content": "Paris."}]
TEXT = ("Le chat dort sur le tapis. La capitale de la France est Paris. "
        "L'été, les élèves partent en vacances à la mer. ") * 60


@pytest.fixture(scope="module")
def tok():
    return train_bpe([TEXT], vocab_size=400, verbose=False)


def _model(**kw):
    torch.manual_seed(0)
    return GPT(config_from_preset("tiny", vocab_size=512, **kw)).eval()


# --------------------------------------------------------------------------- tokenizer

def test_tokenizer_round_trips_and_compresses(tok):
    s = "L'été, le chat dort à Paris. Ça marche ? 12345 €"
    assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2  # merges actually learned
    assert 256 < tok.n_vocab <= 400  # stops early once no pair repeats


def test_special_tokens_cannot_be_typed(tok):
    ids = tok.encode("<|user|> <|endoftext|>")
    assert tok.user not in ids and tok.eot not in ids
    assert tok.decode([tok.eot, *tok.encode("ok")]) == "ok"  # specials never leak into text


def test_tokenizer_survives_json(tok, tmp_path):
    tok.save(tmp_path / "t.json")
    again = Tokenizer.load(tmp_path / "t.json")
    assert again == tok and again.encode(TEXT[:200]) == tok.encode(TEXT[:200])


# --------------------------------------------------------------------------- model

def test_forward_shapes_and_loss():
    m = _model()
    x = torch.randint(0, 500, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, m.cfg.vocab_size)
    assert loss.item() > 0
    last, none = m(x)
    assert last.shape == (2, 1, m.cfg.vocab_size) and none is None


def test_kv_cache_matches_full_forward():
    m = _model()
    x = torch.randint(0, 500, (1, 12))
    with torch.no_grad():
        full, _ = m(x)
        cache = [None] * m.cfg.n_layer
        m(x[:, :8], cache=cache)
        for i in range(8, 12):
            step, _ = m(x[:, i:i + 1], cache=cache, pos=i)
    assert torch.allclose(full[0, -1], step[0, -1], atol=1e-4)


def test_generate_keeps_the_prompt_when_asked_for_many_tokens():
    m = _model()
    prompt = torch.randint(0, 500, (1, 20))
    out = m.generate(prompt, max_new_tokens=500, temperature=0)
    assert torch.equal(out[0, :20], prompt[0])  # prompt never cropped away
    assert out.shape[1] - 20 <= m.cfg.block_size


def test_generate_output_starts_with_the_full_prompt_even_when_too_long():
    m = _model()
    prompt = torch.randint(0, 500, (1, 100))  # longer than block_size (64)
    out = m.generate(prompt, max_new_tokens=5, temperature=0)
    assert torch.equal(out[0, :100], prompt[0]) and out.shape[1] == 105


# --------------------------------------------------------------------------- chat data

def test_chat_template_learns_only_assistant_tokens(tok):
    ids, mask = render_conversation(CONV, tok)
    learned = [t for t, k in zip(ids, mask) if k]
    assert tok.decode(learned[:-1]) == "Paris."
    assert learned[-1] == tok.eot
    assert ids[0] == tok.user and tok.assistant in ids
    assert render_prompt(CONV[:1], tok)[-1] == tok.assistant


@pytest.mark.parametrize("row", [
    {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
    {"conversation": [{"role": "user", "text": "a"}, {"role": "assistant", "text": "b"}], "context": ""},
    {"conversations": [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}]},
    {"instruction": "a", "input": "", "output": "b"},
    {"instruction": "a", "input": "Aucun", "output": "b"},
])
def test_to_messages_normalizes_every_layout(row):
    assert to_messages(row) == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]


def test_identity_conversations_are_valid():
    lines = IDENTITY["fr"].read_text(encoding="utf-8").splitlines()
    convs = [to_messages(json.loads(line)) for line in lines if line]
    assert len(convs) >= 10
    assert all(c[-1]["role"] == "assistant" and c[-1]["content"] for c in convs)
    assert not any("—" in c[-1]["content"] for c in convs)  # no em dash


def test_sft_loader_ignores_user_targets(tok, tmp_path):
    tok.save(tmp_path / "tok.json")
    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join(json.dumps({"messages": CONV}) for _ in range(5)), encoding="utf-8")
    prepare_sft(tmp_path / "sft", tmp_path / "tok.json", jsonl=path, val_every=1000)
    x, y = SFTLoader(tmp_path / "sft", "train", 2, 32).next_batch()
    assert x.shape == y.shape == (2, 32)
    assert (y == -1).any() and (y != -1).any()


# --------------------------------------------------------------------------- training

def test_lr_schedule_warms_up_then_decays():
    assert lr_at(0, 100, 1.0, 10) < lr_at(9, 100, 1.0, 10)
    assert lr_at(100, 100, 1.0, 10) == pytest.approx(0.1)


def test_train_learns_resumes_exports_and_chats(tok, tmp_path):
    from ownai.gpt.chat import OwnGPT

    tok.save(tmp_path / "tok.json")
    docs = []
    for i in range(3):
        docs.append(tmp_path / f"d{i}.txt")
        docs[-1].write_text(TEXT, encoding="utf-8")
    prepare_pretrain(tmp_path / "data", tmp_path / "tok.json", total_tokens=10_000, val_tokens=500,
                     text_files=docs)
    run = tmp_path / "run"
    kw = dict(data=tmp_path / "data", out=run, preset="tiny", batch_size=4, grad_accum=1, warmup=2,
              eval_every=1000, eval_iters=2, save_every=1000, lr=3e-3)
    val = train(max_steps=40, **kw)
    assert val < 3.0  # a repeated paragraph is learned fast
    model, ckpt = load_checkpoint(run / "model.pt")
    assert ckpt["step"] == 40 and isinstance(model, GPT)
    assert tokenizer_of(ckpt) == tok  # the checkpoint carries its own vocabulary
    train(max_steps=45, **kw)  # resumes from ckpt.pt instead of starting over
    log = [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["step"] == 45
    assert np.isfinite(log[-1]["val_loss"]) and 0 < log[-1]["val_bpb"] < 8

    bot = OwnGPT.load(run / "model.pt")
    pieces = list(bot._stream([tok.eot] + tok.encode("Le chat"), 20, 0.0, None))
    assert "".join(pieces) == bot.complete("Le chat", 20, 0.0)
    assert "�" not in "".join(pieces)


def test_repetition_penalty_breaks_loops(tok, tmp_path):
    """A model trained on one repeated sentence loops; the penalty must stop it."""
    from ownai.gpt.chat import OwnGPT

    tok.save(tmp_path / "tok.json")
    docs = [tmp_path / f"d{i}.txt" for i in range(3)]
    for d in docs:
        d.write_text("Le chat dort sur le tapis. " * 300, encoding="utf-8")
    prepare_pretrain(tmp_path / "data", tmp_path / "tok.json", total_tokens=8_000, val_tokens=400,
                     text_files=docs)
    run = tmp_path / "run"
    train(data=tmp_path / "data", out=run, preset="tiny", batch_size=4, grad_accum=1, warmup=2,
          eval_every=1000, eval_iters=2, save_every=1000, lr=3e-3, max_steps=60)
    bot = OwnGPT.load(run / "model.pt")
    ids = [tok.eot] + tok.encode("Le chat")
    looping = "".join(bot._stream(ids, 40, 0.0, None, penalty=0.0))
    penalised = "".join(bot._stream(ids, 40, 0.0, None, penalty=3.0))  # this test model is extreme; a real one needs far less
    assert looping.count("tapis") > 1  # without it, the sentence just repeats
    assert penalised.count("tapis") < looping.count("tapis")
