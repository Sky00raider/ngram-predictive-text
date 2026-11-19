"""Load and clean the headline splits.

`normalize` is copied from Project 2 (malayalam-gpt/src/clean.py) so this
project stays self-contained. It matters here: these headlines were scraped
years ago and often spell chillu letters the old way (consonant + virama +
ZWJ, e.g. "\u0d32\u0d4d\u200d"), while modern text uses the single atomic letter ("\u0d7d").
Both look identical on screen, but a tokenizer sees different code points.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
SPLITS = ("train", "validation", "test")

VIRAMA = "\u0d4d"
ZWJ, ZWNJ, ZWSP = "\u200d", "\u200c", "\u200b"

# consonant + virama + ZWJ  ->  atomic chillu letter
CHILLU = {
    "\u0d23": "\u0d7a",  # ണ -> ൺ
    "\u0d28": "\u0d7b",  # ന -> ൻ
    "\u0d30": "\u0d7c",  # ര -> ർ
    "\u0d32": "\u0d7d",  # ല -> ൽ
    "\u0d33": "\u0d7e",  # ള -> ൾ
    "\u0d15": "\u0d7f",  # ക -> ൿ
}
# Matches both orders: virama+ZWJ (usual) and ZWJ+virama (rare variant).
CHILLU_RE = re.compile(f"([{''.join(CHILLU)}])(?:{VIRAMA}{ZWJ}|{ZWJ}{VIRAMA})")

# Malayalam digits -> ASCII digits, curly quotes -> straight quotes.
CHAR_MAP = {chr(0x0D66 + i): str(i) for i in range(10)}
CHAR_MAP.update({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
CHAR_MAP = str.maketrans(CHAR_MAP)


def normalize(text: str) -> str:
    """NFC, legacy chillus -> atomic chillus, drop ZWJ/ZWNJ/zero-width space, tidy spaces.

    Unlike Project 2 we keep Latin letters and all punctuation: a headline is
    short, and an English word in it ("iPhone", "IPL") is a useful clue.
    """
    text = unicodedata.normalize("NFC", text)
    text = CHILLU_RE.sub(lambda m: CHILLU[m.group(1)], text)
    text = text.replace(ZWJ, "").replace(ZWNJ, "").replace(ZWSP, "").translate(CHAR_MAP)
    return " ".join(text.split())


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """Return [(clean headline, label name), ...] for one split."""
    rows = []
    with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rows.append((normalize(record["text"]), record["label"]))
    return rows


def load_all(data_dir: Path = DATA_DIR) -> dict[str, list[tuple[str, str]]]:
    return {split: load_split(split, data_dir) for split in SPLITS}


def label_names(train_rows: list[tuple[str, str]]) -> list[str]:
    """Labels that actually occur in training, sorted, so the order is stable.

    The IndicGLUE label list is shared by every language (10 names), but each
    language only uses a few of them; we only make label tokens for those.
    """
    return sorted(Counter(label for _, label in train_rows))
