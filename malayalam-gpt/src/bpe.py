"""Byte-pair encoding over Unicode code points (not bytes).

Start from one token per character, then repeatedly merge the most frequent
adjacent pair of tokens into a new token. Frequent Malayalam letter + vowel-sign
combinations and whole common words end up as single tokens.

Text is first cut into chunks (a word with its leading space, a run of digits,
punctuation, whitespace) and merges only happen inside a chunk, so a token
never spans two words.
"""
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

# Chunk = optional leading space + one of: word / digits / punctuation, or a run of whitespace.
# Word characters are "anything that isn't whitespace, a digit or punctuation",
# so Malayalam vowel signs and the virama stay glued to their letters.
_PUNCT = re.escape(string.punctuation + "“”‘’…–—")
CHUNK_RE = re.compile(rf" ?[^\s\d{_PUNCT}]+| ?\d+| ?[{_PUNCT}]+|\s+(?!\S)|\s+")


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every adjacent occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, chars: list[str], merges: list[tuple[int, int]]):
        self.chars = chars
        self.merges = [tuple(m) for m in merges]  # merge k creates token id len(chars) + k
        self.stoi = {c: i for i, c in enumerate(chars)}
        # Every token's text: base chars first, then each merge = its two parts glued together.
        self.itos = dict(enumerate(chars))
        for k, (a, b) in enumerate(self.merges):
            self.itos[len(chars) + k] = self.itos[a] + self.itos[b]
        # pair -> new token id; a smaller id means the merge was learned earlier (higher priority)
        self.ranks = {pair: len(chars) + k for k, pair in enumerate(self.merges)}
        self.cache: dict[str, list[int]] = {}  # chunk -> ids, so repeated words are free

    @classmethod
    def from_text(cls, text: str, vocab_size: int = 1024) -> "BPETokenizer":
        chars = sorted(set(text))
        n_merges = vocab_size - len(chars)

        # Work on unique chunks and how often each occurs, not on the raw stream.
        freqs = Counter(CHUNK_RE.findall(text))
        stoi = {c: i for i, c in enumerate(chars)}
        words = [[stoi[c] for c in w] for w in freqs]
        counts_of = list(freqs.values())

        # pair_counts: how often each adjacent pair occurs (weighted by chunk frequency)
        # where: pair -> set of chunk indices containing it, so a merge only touches those
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for w, (ids, f) in enumerate(zip(words, counts_of)):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += f
                where[pair].add(w)

        merges: list[tuple[int, int]] = []
        for k in range(max(n_merges, 0)):
            if not pair_counts:
                break  # nothing left to merge
            best = max(pair_counts, key=pair_counts.get)
            new_id = len(chars) + k
            merges.append(best)

            # Update only the chunks that contain `best`: remove their old pair counts,
            # merge, then add the new pair counts.
            for w in where.pop(best):
                ids, f = words[w], counts_of[w]
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] -= f
                ids = words[w] = merge_pair(ids, best, new_id)
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += f
                    where[pair].add(w)
            # drop pairs that no longer occur so max() stays fast
            for pair in [p for p, c in pair_counts.items() if c <= 0]:
                del pair_counts[pair]
        return cls(chars, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + len(self.merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self.cache:
            return self.cache[chunk]
        ids = [self.stoi[c] for c in chunk]
        while len(ids) > 1:
            # of the pairs present, apply the one that was learned first
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge_pair(ids, pair, self.ranks[pair])
        self.cache[chunk] = ids
        return ids

    def encode(self, s: str) -> list[int]:
        # characters never seen in training would have no id; fail loudly instead of guessing
        unknown = sorted(set(s) - self.stoi.keys())
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown}")
        out = []
        for chunk in CHUNK_RE.findall(s):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    # Merges are stored as [id, id] pairs in the order they were learned.
    def save(self, path: str | Path) -> None:
        data = {"chars": self.chars, "merges": self.merges}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"], data["merges"])
