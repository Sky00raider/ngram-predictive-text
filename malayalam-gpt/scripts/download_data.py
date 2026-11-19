"""Stream Malayalam text from the Hugging Face Hub into data/raw.txt.

We stream (never download whole dumps) and stop at ~25 MB of UTF-8.
Run from the project root:  python scripts/download_data.py
"""
from pathlib import Path

from datasets import load_dataset

TARGET_BYTES = 25 * 1024 * 1024
OUT_PATH = Path("data/raw.txt")

# Sources to try in order; the first one that works without auth wins.
# Each entry: (name, load_dataset kwargs, text column)
SOURCES = [
    ("sangraha-verified", dict(path="ai4bharat/sangraha", data_dir="verified/mal"), "text"),
    ("indiccorp-v2", dict(path="ai4bharat/IndicCorpV2", name="indiccorp_v2", split="mal_Mlym"), "text"),
    ("culturax", dict(path="uonlp/CulturaX", name="ml"), "text"),
    ("mc4", dict(path="allenai/c4", name="ml"), "text"),
    ("wikipedia", dict(path="wikimedia/wikipedia", name="20231101.ml"), "text"),
]


def paragraphs(text: str):
    """Yield non-empty lines, so each output line is one paragraph."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield line


def write_raw(dataset, column: str) -> int:
    """Write paragraphs until we reach TARGET_BYTES; return bytes written."""
    written = 0
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for row in dataset:
            for para in paragraphs(row[column]):
                f.write(para + "\n")
                written += len(para.encode("utf-8")) + 1
            if written >= TARGET_BYTES:
                break
    return written


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name, kwargs, column in SOURCES:
        try:
            kwargs = {"split": "train", **kwargs}  # an explicit split in kwargs wins
            dataset = load_dataset(streaming=True, **kwargs)
            first = next(iter(dataset))  # fails here if gated / missing
            assert column in first
        except Exception as e:  # gated, renamed, needs auth, ...
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        print(f"Using source: {name}")
        written = write_raw(dataset, column)
        print(f"Wrote {written / 1e6:.1f} MB to {OUT_PATH}")
        return
    raise SystemExit("No source worked.")


if __name__ == "__main__":
    main()
