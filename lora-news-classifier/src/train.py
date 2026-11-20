"""Fine-tune a LoRA adapter (plus the label-token rows) to classify headlines.

    python -m src.train --base distilgpt2 --rank 8 --targets attn
    python -m src.train --base bloom-560m --targets attn+mlp --epochs 2
    python -m src.train --limit 200 --max-steps 30      # quick smoke test

Each run writes to outputs/runs/<name>/:
    config.json   the full Config
    log.csv       step, train loss, validation accuracy
    metrics.json  final validation/test accuracy (full and unseen-only)
    adapter.pt    LoRA weights + label-token rows + config (see src/adapter.py)
    test_logits.pt  test headlines, labels and label logits, to check a reloaded adapter against
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from src.adapter import build_classifier, save_adapter
from src.classify import HeadlineClassifier
from src.config import Config, get_device
from src.data import dedupe, label_names, load_all, unseen_mask
from src.lora import count_parameters

RUNS_DIR = Path("outputs/runs")


def evaluate(clf: HeadlineClassifier, rows, train_rows) -> dict:
    """Accuracy on all rows, on the rows not seen in train, and per class."""
    clf.eval()
    preds = clf.predict([text for text, _ in rows])
    gold = [clf.labels.index(label) for _, label in rows]
    correct = [p == g for p, g in zip(preds, gold)]
    unseen = unseen_mask(rows, train_rows)
    per_class = {}
    for k, label in enumerate(clf.labels):
        hits = [c for c, g in zip(correct, gold) if g == k]
        per_class[label] = sum(hits) / len(hits)
    return {
        "acc": sum(correct) / len(correct),
        "acc_unseen": sum(c for c, u in zip(correct, unseen) if u) / sum(unseen),
        "n": len(rows),
        "n_unseen": sum(unseen),
        "per_class": per_class,
    }


def lr_at(step: int, total: int, warmup: int) -> float:
    """Linear warm-up, then linear decay to 0 (a multiplier on cfg.lr)."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def train(cfg: Config, run_name: str, limit: int | None = None, max_steps: int | None = None,
          eval_every: int | None = None) -> tuple[HeadlineClassifier, dict]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = get_device()

    data = load_all()
    train_rows = dedupe(data["train"])
    cfg.labels = label_names(train_rows)
    if limit:  # smoke tests: a small random slice of every split
        train_rows = random.sample(train_rows, min(limit, len(train_rows)))
        data = {s: random.sample(rows, min(limit, len(rows))) for s, rows in data.items()}

    clf = build_classifier(cfg, device)
    trainable, total = count_parameters(clf)
    print(f"{cfg.base} | targets={cfg.targets} r={cfg.rank} alpha={cfg.alpha} | device={device}")
    print(f"trainable {trainable:,} of {total:,} params ({trainable / total:.3%}) "
          f"| train {len(train_rows)} headlines (deduplicated), labels {cfg.labels}")
    room = cfg.max_len - len(clf.suffix_ids)
    cut = sum(clf.n_tokens(text) > room for text, _ in train_rows) / len(train_rows)
    print(f"max_len {cfg.max_len}: {cut:.1%} of train headlines are truncated")

    steps_per_epoch = math.ceil(len(train_rows) / cfg.batch_size)
    total_steps = min(max_steps or 10**9, steps_per_epoch * cfg.epochs)
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    eval_every = eval_every or steps_per_epoch
    optimizer = torch.optim.AdamW(clf.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s, total_steps, warmup))

    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    log_file = (out_dir / "log.csv").open("w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["step", "train_loss", "val_acc", "val_acc_unseen", "seconds"])

    before = evaluate(clf, data["validation"], train_rows)
    print(f"before training: val acc {before['acc']:.3f}")
    log.writerow([0, "", f"{before['acc']:.4f}", f"{before['acc_unseen']:.4f}", 0])
    log_file.flush()

    step, start, running = 0, time.time(), []
    train_secs = 0.0  # time spent in training steps only (evals excluded)
    while step < total_steps:
        random.shuffle(train_rows)
        for i in range(0, len(train_rows), cfg.batch_size):
            batch = train_rows[i:i + cfg.batch_size]
            targets = torch.tensor([cfg.labels.index(label) for _, label in batch])
            t0 = time.time()
            clf.train()
            loss = clf.loss([text for text, _ in batch], targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running.append(loss.item())
            train_secs += time.time() - t0
            step += 1
            if step % 50 == 0 and step % eval_every != 0:  # cheap progress line, no eval
                print(f"step {step:5d}/{total_steps} | loss (last 50) {sum(running[-50:]) / len(running[-50:]):.4f} "
                      f"| {train_secs / step:.2f} s/train step")

            if step % eval_every == 0 or step == total_steps:
                val = evaluate(clf, data["validation"], train_rows)
                secs = time.time() - start
                mean_loss = sum(running) / len(running)
                print(f"step {step:5d}/{total_steps} | loss {mean_loss:.4f} | val acc {val['acc']:.3f} "
                      f"(unseen {val['acc_unseen']:.3f}) | {train_secs / step:.2f} s/train step")
                log.writerow([step, f"{mean_loss:.4f}", f"{val['acc']:.4f}", f"{val['acc_unseen']:.4f}", f"{secs:.0f}"])
                log_file.flush()
                running = []
            if step >= total_steps:
                break
    log_file.close()

    metrics = {
        "base": cfg.base, "targets": cfg.targets, "rank": cfg.rank,
        "trainable_params": trainable, "steps": step,
        "seconds": round(time.time() - start),
        "seconds_per_step": round(train_secs / step, 3),
        "validation": evaluate(clf, data["validation"], train_rows),
        "test": evaluate(clf, data["test"], train_rows),
    }
    clf.eval()
    test_logits = torch.cat([clf.label_logits([text for text, _ in data["test"][i:i + 64]]).cpu()
                             for i in range(0, len(data["test"]), 64)])
    torch.save({"texts": [text for text, _ in data["test"]],
                "labels": [label for _, label in data["test"]],
                "logits": test_logits}, out_dir / "test_logits.pt")
    metrics["adapter_bytes"] = save_adapter(clf, cfg, out_dir / "adapter.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    t = metrics["test"]
    print(f"test acc {t['acc']:.3f} (n={t['n']}) | unseen-only {t['acc_unseen']:.3f} (n={t['n_unseen']})")
    return clf, metrics


def parse_args() -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    p.add_argument("--base", default=defaults.base, choices=["distilgpt2", "bloom-560m"])
    p.add_argument("--targets", default=defaults.targets, choices=["attn", "attn+mlp"])
    p.add_argument("--rank", type=int, default=defaults.rank)
    p.add_argument("--alpha", type=float, default=None, help="default: 2 * rank")
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--max-len", type=int, default=None, help="default: set per base model in config.py")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--name", default=None, help="run folder name (default: built from the settings)")
    p.add_argument("--limit", type=int, default=None, help="use only this many headlines per split")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None, help="steps between evals (default: once per epoch)")
    args = p.parse_args()
    cfg = Config(base=args.base, targets=args.targets, rank=args.rank,
                 alpha=args.alpha if args.alpha is not None else 2.0 * args.rank,
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 max_len=args.max_len, seed=args.seed)
    return cfg, args


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress live, even when piped
    cfg, args = parse_args()
    name = args.name or f"{cfg.base}_{cfg.targets}_r{cfg.rank}"
    train(cfg, name, limit=args.limit, max_steps=args.max_steps, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
