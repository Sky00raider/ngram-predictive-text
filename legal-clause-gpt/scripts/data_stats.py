"""Dataset statistics: class imbalance, duplicates, clause length in GPT-2 tokens.

Answers the questions that set the run's knobs:
  * how imbalanced is LEDGAR, and what does a stratified subset leave the
    rarest classes with?  (train_size, min_per_class)
  * how long are clauses in GPT-2 tokens, and what share fits in max_len?
  * how much boilerplate repeats, within train and between train and test?

Writes outputs/data_stats.txt (and prints it).
Run from the project root:  python scripts/data_stats.py [--train-size 20000]   (size of the subset to describe)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config  # noqa: E402
from src.data import class_counts, dedupe, load_labels, load_split, seen_mask, stratified_subset  # noqa: E402

OUT = Path("outputs/data_stats.txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-size", type=int, default=Config().train_size)
    a = p.parse_args()
    cfg = Config(train_size=a.train_size)

    labels = load_labels()
    splits = {s: load_split(s) for s in ("train", "validation", "test")}
    lines = []
    say = lines.append

    say("## Splits")
    for s, rows in splits.items():
        say(f"{s:>10}: {len(rows):6d} clauses")

    say("\n## Duplicates (exact, after lower-casing and collapsing whitespace)")
    train_dedup, report = dedupe(splits["train"])
    say(f"train: {report['repeats_dropped']} repeats dropped -> {report['kept']} unique; "
        f"{report['texts_with_conflicting_labels']} texts appear with more than one label")
    for s in ("validation", "test"):
        seen = np.mean(seen_mask(splits[s], splits["train"]))
        say(f"{s}: {seen:.1%} of clauses also occur verbatim in the full train split")

    say("\n## Class imbalance")
    full = np.array(class_counts(train_dedup, labels))
    subset = stratified_subset(train_dedup, cfg.train_size, cfg.min_per_class, cfg.seed)
    sub = np.array(class_counts(subset, labels))
    test = np.array(class_counts(splits["test"], labels))
    say(f"deduplicated train: min {full.min()}  median {int(np.median(full))}  max {full.max()}  "
        f"(largest/smallest = {full.max() / full.min():.0f}x)")
    say(f"{len(subset)}-clause subset: min {sub.min()}  median {int(np.median(sub))}  max {sub.max()}; "
        f"{(sub < 20).sum()} classes under 20 examples, {(sub < 50).sum()} under 50")
    say(f"test: min {test.min()}  median {int(np.median(test))}  max {test.max()}; "
        f"{(test < 10).sum()} classes with fewer than 10 test clauses")
    order = np.argsort(-full)
    say("\n  rank  label                                 full  subset  test")
    for r in list(range(10)) + ["..."] + list(range(len(labels) - 10, len(labels))):
        if r == "...":
            say("   ...")
            continue
        i = order[r]
        say(f"  {r + 1:4d}  {labels[i][:36]:36s} {full[i]:5d}  {sub[i]:6d}  {test[i]:4d}")

    say("\n## Clause length in GPT-2 tokens (clause only, before the prompt suffix)")
    tok = AutoTokenizer.from_pretrained(cfg.hf_name)
    for s, rows in splits.items():
        n = np.array([len(x) for x in tok([r["text"] for r in rows], add_special_tokens=False,
                                          verbose=False).input_ids])
        q = np.percentile(n, [50, 90, 95, 99])
        fits = "  ".join(f"<= {m}: {np.mean(n <= m - 5):.1%}" for m in (128, 256, 384, 512))
        say(f"{s:>10}: mean {n.mean():.0f}  median {q[0]:.0f}  p90 {q[1]:.0f}  p95 {q[2]:.0f}  "
            f"p99 {q[3]:.0f}  max {n.max()}")
        say(f"{'':>10}  fits uncut in max_len {fits}")

    text = "\n".join(lines)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
