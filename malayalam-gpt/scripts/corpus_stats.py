"""Print (and save) statistics about data/corpus.txt.

Run from the project root:  python scripts/corpus_stats.py
"""
import unicodedata
from collections import Counter
from pathlib import Path

CORPUS_PATH = Path("data/corpus.txt")
OUT_PATH = Path("outputs/corpus_stats.txt")
SAMPLE_CHARS = 600


def show(ch: str) -> str:
    """Printable form of a character (newline and space made visible)."""
    return {"\n": "\\n", " ": "' '"}.get(ch, ch)


def build_report(text: str) -> str:
    counts = Counter(text)
    lines = [
        f"size: {len(text.encode('utf-8')) / 1e6:.2f} MB",
        f"characters: {len(text)}",
        f"lines: {text.count(chr(10))}",
        f"vocab size: {len(counts)}",
        "",
        "char  code    name                                          count",
    ]
    for ch in sorted(counts):
        name = "LINE FEED" if ch == "\n" else unicodedata.name(ch, "<unnamed>")
        lines.append(f"{show(ch):<5} U+{ord(ch):04X}  {name:<44} {counts[ch]}")
    mid = len(text) // 2
    lines += ["", f"sample (~{SAMPLE_CHARS} chars from the middle):", text[mid:mid + SAMPLE_CHARS]]
    return "\n".join(lines)


def main():
    report = build_report(CORPUS_PATH.read_text(encoding="utf-8"))
    print(report)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
