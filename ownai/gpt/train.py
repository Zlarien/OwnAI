"""Training loop for OwnGPT, built to survive free GPUs.

Colab and Kaggle sessions end without warning, so everything here is resumable:
the run directory holds ``ckpt.pt`` (weights + optimizer + step + tokenizer),
and relaunching the same command picks up exactly where it stopped. ``--hours``
stops cleanly before the session limit and saves.

Stages:
  * pretrain : learn language from raw text (next-token prediction).
  * sft      : start from a pretrained model (``--init``) and learn to chat.

Besides the loss, validation reports bits per byte (bpb): the loss converted to
bits and divided by the UTF-8 bytes it covers. Unlike the loss it does not depend
on the tokenizer, so runs with different vocabularies stay comparable.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from ownai.gpt.data import PretrainLoader, SFTLoader, render_prompt
from ownai.gpt.model import GPT, GPTConfig, config_from_preset
from ownai.gpt.tokenizer import Tokenizer

STAGE_DEFAULTS = {
    "pretrain": dict(lr=6e-4, warmup=500, weight_decay=0.1),
    "sft": dict(lr=5e-5, warmup=50, weight_decay=0.0),
}

SAMPLE_PROMPTS = {
    "fr": {"pretrain": "La Révolution française", "sft": "Quelle est la capitale de la France ?"},
    "en": {"pretrain": "The most important thing about learning is", "sft": "What is the capital of France?"},
}


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path, device="cpu"):
    """Rebuild a GPT from ``ckpt.pt`` or an exported ``model.pt`` (tokenizer in ``ckpt``)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


def tokenizer_of(ckpt) -> Tokenizer:
    return Tokenizer(ckpt["tokenizer"])


def _save(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the last good checkpoint


def lr_at(step: int, max_steps: int, lr: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, loader, iters: int, ctx, token_bytes):
    """Mean validation loss and bits per byte."""
    model.eval()
    nats, tokens, n_bytes = 0.0, 0, 0
    for _ in range(iters):
        x, y = loader.next_batch()
        with ctx:
            _, loss = model(x, y)
        valid = y[y != -1]
        nats += loss.item() * valid.numel()
        tokens += valid.numel()
        n_bytes += int(token_bytes[valid].sum())
    model.train()
    return nats / max(1, tokens), nats / math.log(2) / max(1, n_bytes)


@torch.no_grad()
def sample_text(model, tok: Tokenizer, stage: str, device, ctx, max_new_tokens: int = 60) -> str:
    model.eval()
    prompt = SAMPLE_PROMPTS.get(tok.lang, SAMPLE_PROMPTS["en"])[stage]
    if stage == "sft":
        ids = render_prompt([{"role": "user", "content": prompt}], tok)
    else:
        ids = [tok.eot] + tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with ctx:
        new = list(model.stream(idx, max_new_tokens, temperature=0.8, stop_token=tok.eot,
                                generator=torch.Generator(device=device).manual_seed(0)))
    model.train()
    return (prompt + " -> " if stage == "sft" else prompt) + tok.decode(new)


def train(stage="pretrain", data="data/owngpt/pretrain", out="runs/pretrain", preset="t4",
          init=None, max_steps=5000, batch_size=16, grad_accum=4, lr=None, warmup=None,
          weight_decay=None, eval_every=250, eval_iters=20, save_every=500, log_every=10,
          hours=None, compile=False, device=None, seed=0):
    defaults = STAGE_DEFAULTS[stage]
    lr = lr if lr is not None else defaults["lr"]
    warmup = warmup if warmup is not None else defaults["warmup"]
    weight_decay = weight_decay if weight_decay is not None else defaults["weight_decay"]

    device = pick_device(device)
    cuda = device.startswith("cuda")
    if cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype = torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if cuda else nullcontext()
    torch.manual_seed(seed)

    tok = Tokenizer.load(Path(data) / "tokenizer.json")
    run = Path(out)
    run.mkdir(parents=True, exist_ok=True)
    ckpt_path = run / "ckpt.pt"
    start_step, opt_state = 0, None
    if ckpt_path.exists():
        model, ckpt = load_checkpoint(ckpt_path, device)
        start_step, opt_state = ckpt["step"], ckpt.get("optimizer")
        print(f"resuming {stage} from step {start_step}", flush=True)
    elif init:
        model, ckpt = load_checkpoint(init, device)
        if tokenizer_of(ckpt) != tok:
            raise ValueError(f"{init} was trained with another tokenizer than {data}/tokenizer.json")
        model.float()
        print(f"initialised from {init}", flush=True)
    else:
        vocab = 64 * math.ceil(tok.n_vocab / 64)  # round up: matmuls prefer multiples of 64
        model = GPT(config_from_preset(preset, vocab_size=vocab)).to(device)
    cfg = model.cfg
    print(f"OwnGPT {cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}d: {model.num_params() / 1e6:.1f}M params "
          f"on {device} ({str(dtype).split('.')[-1]})", flush=True)

    opt = model.configure_optimizer(lr, weight_decay)
    if opt_state:
        opt.load_state_dict(opt_state)
    fwd = torch.compile(model) if compile else model
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and dtype == torch.float16)

    Loader = SFTLoader if stage == "sft" else PretrainLoader
    train_loader = Loader(data, "train", batch_size, cfg.block_size, device, seed=seed + start_step)
    val_loader = Loader(data, "val", batch_size, cfg.block_size, device, seed=1234)
    token_bytes = torch.from_numpy(tok.token_bytes()).to(device)
    tokens_per_step = batch_size * cfg.block_size * grad_accum
    print(f"{tokens_per_step:,} tokens per step, {max_steps} steps "
          f"= {tokens_per_step * max_steps / 1e9:.2f}B tokens", flush=True)

    def payload(step, weights, optimizer=None):
        p = {"model": weights, "config": cfg.to_dict(), "step": step, "stage": stage,
             "tokenizer": tok.to_dict()}
        if optimizer is not None:
            p["optimizer"] = optimizer
        return p

    def save(step):
        _save(ckpt_path, payload(step, model.state_dict(), opt.state_dict()))

    log = open(run / "log.jsonl", "a", encoding="utf-8")

    def evaluate(step, final=False):
        val, bpb = estimate_loss(fwd, val_loader, eval_iters, ctx, token_bytes)
        text = sample_text(model, tok, stage, device, ctx)
        print(f"step {step} | val loss {val:.3f} | val bpb {bpb:.3f} | {text!r}", flush=True)
        log.write(json.dumps({"step": step, "val_loss": val, "val_bpb": bpb, "sample": text,
                              **({"final": True} if final else {})}, ensure_ascii=False) + "\n")
        log.flush()
        return val

    t_start = t_log = time.time()
    step = start_step
    while step < max_steps:
        if step % eval_every == 0 and step > start_step:
            evaluate(step)
        if step % save_every == 0 and step > start_step:
            save(step)

        for group in opt.param_groups:
            group["lr"] = lr_at(step, max_steps, lr, warmup)
        loss_sum = 0.0
        for _ in range(grad_accum):
            x, y = train_loader.next_batch()
            with ctx:
                _, loss = fwd(x, y)
            scaler.scale(loss / grad_accum).backward()
            loss_sum += loss.item() / grad_accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0:
            now = time.time()
            tok_s = tokens_per_step * log_every / (now - t_log)
            t_log = now
            left = (max_steps - step) * tokens_per_step / tok_s / 3600
            print(f"step {step}/{max_steps} | loss {loss_sum:.3f} | lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {tok_s:,.0f} tok/s | ~{left:.1f}h left", flush=True)
            log.write(json.dumps({"step": step, "loss": loss_sum, "tok_s": tok_s}) + "\n")
        if hours and time.time() - t_start > hours * 3600:
            print(f"time budget of {hours}h reached, saving and stopping", flush=True)
            break

    save(step)
    val = evaluate(step, final=True)
    log.close()
    if step >= max_steps:  # light weights-only export for chatting / the web UI
        _save(run / "model.pt", payload(step, {k: v.half() for k, v in model.state_dict().items()}))
        print(f"exported {run / 'model.pt'}", flush=True)
    return val
