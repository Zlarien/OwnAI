"""Tests for the from-scratch byte-level BPE tokenizer."""
import json

from ownai.tokenizer import BPETokenizer

CORPUS = [
    "Mario jumps over the goomba.",
    "Mario runs and Mario jumps again.",
    "The goomba walks. Mario wins the level.",
] * 20


def test_train_reaches_target_vocab():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    assert tok.vocab_size == 300


def test_encode_decode_roundtrip_is_lossless():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    for text in ["Mario jumps.", "unseen wörds 123!", ""]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_merges_shrink_token_count():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400)
    text = "Mario jumps over the goomba."
    # A trained BPE must encode a familiar sentence in fewer tokens than raw bytes.
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_special_tokens_have_stable_ids():
    tok = BPETokenizer(special_tokens=["<|bos|>", "<|sep|>", "<|eos|>"])
    tok.train(CORPUS, vocab_size=300)
    assert tok.token_to_id("<|bos|>") == 256
    assert tok.token_to_id("<|sep|>") == 257
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.token_to_id("<|bos|>")
    assert ids[-1] == tok.token_to_id("<|eos|>")


def test_save_and_load_roundtrip(tmp_path):
    tok = BPETokenizer(special_tokens=["<|bos|>"])
    tok.train(CORPUS, vocab_size=320)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    text = "Mario jumps over the goomba."
    assert loaded.encode(text) == tok.encode(text)
    # Saved file is human-readable JSON.
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    assert "merges" in blob and "special_tokens" in blob


def test_decode_of_arbitrary_ids_does_not_crash():
    # A generative model emits arbitrary token ids whose bytes may not form
    # valid UTF-8. Decoding must degrade gracefully, never raise.
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    weird_ids = [0xD7, 0xFF, 5, 200, 0x80, 42]
    out = tok.decode(weird_ids)
    assert isinstance(out, str)


def test_special_token_not_split():
    tok = BPETokenizer(special_tokens=["<|bos|>"])
    tok.train(CORPUS, vocab_size=300)
    ids = tok.encode("<|bos|>Mario")
    assert ids[0] == tok.token_to_id("<|bos|>")
    assert tok.decode(ids) == "<|bos|>Mario"
