"""Character-level tokenizer: every Unicode code point is one token.

For Malayalam that means vowel signs and the virama are tokens of their own,
so one visible letter like ക്ഷേ is 4 tokens (ക ് ഷ േ).
"""
import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}  # string -> id
        self.itos = {i: c for i, c in enumerate(chars)}  # id -> string

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Vocabulary = every distinct character in the corpus, sorted."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        # characters never seen in training would have no id; fail loudly instead of guessing
        unknown = sorted({c for c in s if c not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown}")
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    # Saved next to the checkpoint, so generation doesn't need the corpus.
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])
