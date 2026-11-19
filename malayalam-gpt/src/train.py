"""Train the GPT on the Malayalam corpus.

    python -m src.train                          # char-level, size picked for your device
    python -m src.train --max-iters 200          # quick smoke test
    python -m src.train --tokenizer bpe --run bpe

Writes to outputs/<run>/: loss.csv, samples.txt, model.pt, vocab.json, config.json
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from .config import default_config, get_device
from .data import get_batch, load_corpus, train_val_split
from .model import GPT, count_params
from .tokenizer import CharTokenizer

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
SAMPLE_PROMPT = "കേരളത്തിൽ"  # "In Kerala": the same prompt at every checkpoint, so samples are comparable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="char", help="output folder name under outputs/")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--bpe-vocab", default=str(OUTPUTS / "bpe_vocab.json"))
    # any Config field can be overridden from the command line
    for name in ["block_size", "n_embd", "n_heads", "n_layers", "batch_size", "max_iters", "eval_interval", "eval_iters"]:
        p.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ["dropout", "lr"]:
        p.add_argument("--" + name, type=float)
    return p.parse_args()


def build_tokenizer(args, text):
    if args.tokenizer == "char":
        return CharTokenizer.from_text(text)
    from .bpe import BPETokenizer  # trained beforehand by scripts/train_bpe.py
    return BPETokenizer.load(args.bpe_vocab)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    """Average loss over eval_iters random batches of each split (less noisy than one batch)."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, cfg.block_size, cfg.batch_size, device))[1].item() for _ in range(cfg.eval_iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def sample(model, tok, device, n_tokens=200):
    idx = torch.tensor([tok.encode(SAMPLE_PROMPT)], dtype=torch.long, device=device)
    model.eval()
    out = tok.decode(model.generate(idx, n_tokens, temperature=0.8, top_k=40)[0].tolist())
    model.train()
    return out


def main():
    args = parse_args()
    device = get_device()
    cfg = default_config(device)
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    torch.manual_seed(cfg.seed)

    out_dir = OUTPUTS / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    # data -> token ids -> train/val tensors
    text = load_corpus()
    tok = build_tokenizer(args, text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(data)
    splits = {"train": train_data, "val": val_data}

    model = GPT(cfg, tok.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    print(f"device={device} tokenizer={args.tokenizer} vocab={tok.vocab_size} tokens={len(data):,} params={count_params(model):,}")
    print(f"config: {cfg}")
    print(f"untrained loss should be about ln(vocab) = {math.log(tok.vocab_size):.3f}")

    tok.save(out_dir / "vocab.json")
    (out_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "tokenizer": args.tokenizer}, indent=2))

    # samples at the start, early, middle and end show the model improving
    sample_steps = {0, 500, cfg.max_iters // 2, cfg.max_iters}
    log = open(out_dir / "loss.csv", "w", newline="")
    writer = csv.writer(log); writer.writerow(["step", "train_loss", "val_loss", "elapsed_s"])
    samples = open(out_dir / "samples.txt", "w", encoding="utf-8")
    best_val, t0 = float("inf"), time.time()

    for step in range(cfg.max_iters + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(model, splits, cfg, device)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed / 60:.1f} min", flush=True)
            writer.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{elapsed:.0f}"]); log.flush()
            if losses["val"] < best_val:  # keep the checkpoint that generalises best
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "tokenizer": args.tokenizer,
                            "step": step, "val_loss": best_val}, out_dir / "model.pt")
        if step in sample_steps:
            samples.write(f"=== step {step} (val loss {losses['val']:.3f}) ===\n{sample(model, tok, device)}\n\n"); samples.flush()
        if step == cfg.max_iters:
            break

        # one optimisation step: forward, next-token loss, backprop, update
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    log.close(); samples.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val loss {best_val:.4f} -> {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
