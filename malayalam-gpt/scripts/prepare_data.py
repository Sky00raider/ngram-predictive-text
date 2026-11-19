"""Clean data/raw.txt into data/corpus.txt (5-10 MB target).

Run from the project root:  python scripts/prepare_data.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `src` imports work

from src.clean import clean_text

RAW_PATH = Path("data/raw.txt")
OUT_PATH = Path("data/corpus.txt")
MAX_BYTES = 8_000_000
SEED = 0  # paragraph shuffle, so the 8 MB slice mixes topics instead of keeping the first stretch of the stream


def truncate_at_line(text: str, max_bytes: int) -> str:
    """Cut to at most max_bytes of UTF-8, keeping only whole lines."""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    kept, size = [], 0
    for line in text.splitlines(keepends=True):
        size += len(line.encode("utf-8"))
        if size > max_bytes:
            break
        kept.append(line)
    return "".join(kept)


def main():
    raw = RAW_PATH.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    paragraphs = cleaned.splitlines()
    random.Random(SEED).shuffle(paragraphs)  # one paragraph per line, so shuffling keeps each one intact
    cleaned = "\n".join(paragraphs) + "\n"
    corpus = truncate_at_line(cleaned, MAX_BYTES)
    OUT_PATH.write_text(corpus, encoding="utf-8", newline="\n")
    mb = lambda s: len(s.encode("utf-8")) / 1e6
    print(f"raw: {mb(raw):.1f} MB, {raw.count(chr(10))} lines")
    print(f"cleaned: {mb(cleaned):.1f} MB -> corpus: {mb(corpus):.1f} MB, {corpus.count(chr(10))} lines")


if __name__ == "__main__":
    main()
