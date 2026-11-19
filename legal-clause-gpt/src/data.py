"""Loading LEDGAR, deduplication and the stratified training subset.

LEDGAR is heavily imbalanced: the commonest clause types have thousands of
examples, the rarest a few dozen. We train on a subset (see Config.train_size)
drawn *stratified* by class, so the subset keeps the natural imbalance. That
matters for the rare-class analysis: per-class accuracy is plotted against how
often the class appeared in training, and an artificially balanced subset
would hide exactly the effect we want to measure. The one exception is a small
floor (min_per_class) so no class ends up with only one or two examples.

Contracts reuse boilerplate, so the same clause text can occur many times.
Before sampling we drop exact repeats (after whitespace/case normalisation) so
a repeated clause isn't over-weighted, and at evaluation we report accuracy on
test clauses the model never saw verbatim, alongside the overall number.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")


def load_labels(data_dir: Path = DATA_DIR) -> list[str]:
    return json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[dict]:
    """[{"text", "label"}] for one split, label as its name."""
    with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize(text: str) -> str:
    """Key for duplicate detection: lower-case, whitespace collapsed."""
    return re.sub(r"\s+", " ", text).strip().lower()


def dedupe(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep the first occurrence of each normalised text.

    Returns the kept rows and a small report: how many rows were exact repeats,
    and how many distinct texts appeared with more than one label (those keep
    the label of their first occurrence).
    """
    first_label: dict[str, str] = {}
    labels_seen: dict[str, set] = defaultdict(set)
    kept = []
    for row in rows:
        key = normalize(row["text"])
        labels_seen[key].add(row["label"])
        if key not in first_label:
            first_label[key] = row["label"]
            kept.append(row)
    report = {
        "rows": len(rows),
        "kept": len(kept),
        "repeats_dropped": len(rows) - len(kept),
        "texts_with_conflicting_labels": sum(len(s) > 1 for s in labels_seen.values()),
    }
    return kept, report


def stratified_subset(rows: list[dict], size: int, min_per_class: int = 10, seed: int = 0) -> list[dict]:
    """About `size` rows, each class keeping its share of the original.

    Class c with n_c of N rows gets round(n_c * size / N) rows, but at least
    min(n_c, min_per_class). Because of that floor the result can be a few
    rows larger than `size`. If size >= len(rows) everything is returned.
    The output is shuffled; the same seed always gives the same subset.
    """
    rng = random.Random(seed)
    if size >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)
    out = []
    for label in sorted(by_class):
        members = by_class[label]
        take = max(round(len(members) * size / len(rows)), min(len(members), min_per_class))
        out.extend(rng.sample(members, min(take, len(members))))
    rng.shuffle(out)
    return out


def class_counts(rows: list[dict], labels: list[str]) -> list[int]:
    """Count per class, in label-index order (zeros included)."""
    counts = Counter(row["label"] for row in rows)
    return [counts.get(name, 0) for name in labels]


def seen_mask(rows: list[dict], train_rows: list[dict]) -> list[bool]:
    """True where a row's normalised text occurs in train_rows."""
    train_keys = {normalize(r["text"]) for r in train_rows}
    return [normalize(r["text"]) in train_keys for r in rows]
