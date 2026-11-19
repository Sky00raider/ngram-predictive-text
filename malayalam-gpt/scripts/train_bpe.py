"""Train the BPE tokenizer on the training split and save it.

Run from the project root:  python scripts/train_bpe.py
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bpe import BPETokenizer
from src.data import load_corpus

VOCAB_SIZE = 1024
VOCAB_PATH = Path("outputs/bpe_vocab.json")
REPORT_PATH = Path("outputs/bpe_report.txt")


def show(s: str) -> str:
    """Make spaces and newlines visible."""
    return s.replace("\n", "\n").replace(" ", "_")


def main():
    text = load_corpus()
    train_text = text[: int(0.9 * len(text))]  # same 90% cut as train_val_split

    start = time.time()
    tok = BPETokenizer.from_text(train_text, VOCAB_SIZE)
    elapsed = time.time() - start
    tok.save(VOCAB_PATH)

    # How often is each learned token used when the training text is encoded?
    usage = Counter(tok.encode(train_text))
    n_base = len(tok.chars)
    learned = [(i, c) for i, c in usage.items() if i >= n_base]
    learned.sort(key=lambda x: -x[1])

    lines = [
        f"trained on {len(train_text):,} characters in {elapsed:.1f} s",
        f"base characters: {n_base}, merges: {len(tok.merges)}, vocab size: {tok.vocab_size}",
        f"training text encodes to {sum(usage.values()):,} tokens "
        f"({len(train_text) / sum(usage.values()):.2f} chars/token)",
        "(spaces shown as _)",
        "",
        "30 most-used learned tokens:",
    ]
    for i, c in learned[:30]:
        lines.append(f"  {show(tok.itos[i]):<16} {c:>8,}")
    lines += ["", "first 10 merges:"]
    for k in range(10):
        a, b = tok.merges[k]
        lines.append(f"  {k:>3}: {show(tok.itos[a])} + {show(tok.itos[b])} -> {show(tok.itos[n_base + k])}")
    lines += ["", "last 10 merges:"]
    for k in range(len(tok.merges) - 10, len(tok.merges)):
        a, b = tok.merges[k]
        lines.append(f"  {k:>3}: {show(tok.itos[a])} + {show(tok.itos[b])} -> {show(tok.itos[n_base + k])}")

    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
