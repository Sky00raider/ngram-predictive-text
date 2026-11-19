"""Describe the headline dataset and write outputs/data_stats.md.

Reports, per split: class counts, headline length, how many headlines the
cleaning step changed, and how many test/validation headlines also appear
word-for-word in train (leakage would inflate accuracy).
Run from the project root:  python scripts/data_stats.py
"""
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DATA_DIR, SPLITS, ZWJ, label_names, normalize

OUT_PATH = Path("outputs/data_stats.md")


def raw_rows(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    raw = {s: raw_rows(s) for s in SPLITS}
    clean = {s: [(normalize(r["text"]), r["label"]) for r in raw[s]] for s in SPLITS}
    labels = label_names(clean["train"])
    train_texts = {t for t, _ in clean["train"]}

    lines = ["# Dataset: iNLTK Malayalam headlines (IndicGLUE `inltkh.ml`)", ""]

    lines += ["## Class counts", "",
              "| split | " + " | ".join(labels) + " | total |",
              "|---|" + "---|" * (len(labels) + 1)]
    for s in SPLITS:
        counts = Counter(label for _, label in clean[s])
        unknown = set(counts) - set(labels)
        assert not unknown, f"{s} has labels not seen in train: {unknown}"
        cells = [f"{counts[l]} ({counts[l] / len(clean[s]):.0%})" for l in labels]
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {len(clean[s])} |")

    lines += ["", "## Headlines", "",
              "| split | mean chars | median chars | max chars | had legacy ZWJ chillus | "
              "changed by cleaning | duplicates within split | also in train |",
              "|---|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        lens = [len(t) for t, _ in clean[s]]
        zwj = sum(ZWJ in r["text"] for r in raw[s])
        changed = sum(normalize(r["text"]) != r["text"] for r in raw[s])
        texts = [t for t, _ in clean[s]]
        dups = len(texts) - len(set(texts))
        in_train = "-" if s == "train" else str(sum(t in train_texts for t in texts))
        lines.append(f"| {s} | {mean(lens):.1f} | {median(lens):.0f} | {max(lens)} | "
                     f"{zwj} | {changed} | {dups} | {in_train} |")

    lines += ["", "## One example per class (train, after cleaning)", ""]
    for label in labels:
        text = next(t for t, l in clean["train"] if l == label)
        lines.append(f"- **{label}**: {text}")

    report = "\n".join(lines) + "\n"
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
