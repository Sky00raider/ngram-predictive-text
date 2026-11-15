"""n-gram language model: counts -> probabilities."""
from collections import Counter, defaultdict

from .text import BOS, EOS, to_sentences, to_tokens


def build(sents, n):
    """context (n-1 words) -> Counter of the words that follow it."""
    foll = defaultdict(Counter)
    for s in sents:
        toks = [BOS] * (n - 1) + to_tokens(s) + [EOS]
        for i in range(len(toks) - n + 1):
            foll[tuple(toks[i : i + n - 1])][toks[i + n - 1]] += 1
    return foll


class NgramLM:
    """An n-gram language model: fit -> prob / suggest_next."""

    def __init__(self, n=2):
        self.n = n

    def fit(self, text):
        self.sents = to_sentences(text)
        self.foll = build(self.sents, self.n)
        return self

    def prob(self, ctx, w):
        """Maximum-likelihood P(w | ctx) = count(ctx, w) / count(ctx)."""
        c = self.foll.get(tuple(ctx), Counter())
        total = sum(c.values())
        return c[w] / total if total else 0.0

    def _context(self, text):
        padded = [BOS] * (self.n - 1) + to_tokens(text)
        return tuple(padded[len(padded) - (self.n - 1) :])

    def suggest_next(self, text, topk=3):
        c = self.foll.get(self._context(text), Counter())
        words = [w for w, _ in c.most_common() if w != EOS]
        return words[:topk]
