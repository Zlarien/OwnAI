"""Talk to a trained OwnGPT checkpoint, in the terminal or from the web UI."""
from __future__ import annotations

import torch

from ownai.gpt.data import render_prompt
from ownai.gpt.train import load_checkpoint, pick_device, tokenizer_of

# A 42M model loops on its own words; taking points off tokens it just used
# keeps answers moving without changing anything about the trained weights.
REPETITION_PENALTY = 0.4


class OwnGPT:
    def __init__(self, model, tok, device: str, stage: str = "sft"):
        self.model, self.tok, self.device, self.stage = model.eval(), tok, device, stage

    @classmethod
    def load(cls, path, device: str | None = None):
        device = pick_device(device)
        model, ckpt = load_checkpoint(path, device)
        model.float()  # exported weights are fp16; CPU inference wants fp32
        return cls(model, tokenizer_of(ckpt), device, ckpt.get("stage", "sft"))

    def _stream(self, ids, max_new_tokens, temperature, seed, penalty=REPETITION_PENALTY):
        """Yield decoded text pieces as tokens arrive, never splitting a UTF-8 character."""
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        new, sent = [], ""
        for token in self.model.stream(idx, max_new_tokens, temperature=temperature,
                                       stop_token=self.tok.eot, generator=gen,
                                       repetition_penalty=penalty):
            if token >= len(self.tok.vocab):  # end of text, or a new turn the model tried to open
                break
            new.append(token)
            text = self.tok.decode(new)
            if not text.endswith("�"):  # wait for the rest of a multi-byte character
                yield text[len(sent):]
                sent = text

    def stream_reply(self, messages, max_new_tokens: int = 300, temperature: float = 0.7, seed=None):
        """Stream the next assistant message for a list of {"role", "content"} messages."""
        yield from self._stream(render_prompt(messages, self.tok), max_new_tokens, temperature, seed)

    def reply(self, messages, max_new_tokens: int = 300, temperature: float = 0.7, seed=None) -> str:
        return "".join(self.stream_reply(messages, max_new_tokens, temperature, seed)).strip()

    def complete(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, seed=None) -> str:
        ids = [self.tok.eot] + self.tok.encode(prompt)
        return "".join(self._stream(ids, max_new_tokens, temperature, seed))


def main(path: str, device: str | None = None):
    bot = OwnGPT.load(path, device)
    print(f"OwnGPT ({bot.stage}) prêt. Ligne vide ou Ctrl+C pour quitter.")
    history = []
    try:
        while True:
            q = input("\ntoi > ").strip()
            if not q:
                break
            print("owngpt > ", end="", flush=True)
            if bot.stage == "sft":
                history.append({"role": "user", "content": q})
                parts = []
                for piece in bot.stream_reply(history):
                    parts.append(piece)
                    print(piece, end="", flush=True)
                history.append({"role": "assistant", "content": "".join(parts).strip()})
            else:  # a pretrained-only model just continues text
                for piece in bot._stream([bot.tok.eot] + bot.tok.encode(q), 200, 0.8, None):
                    print(piece, end="", flush=True)
            print()
    except (KeyboardInterrupt, EOFError):
        pass
