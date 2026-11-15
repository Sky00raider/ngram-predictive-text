"""Smoothed n-gram language model: counts -> probabilities."""
from collections import Counter, defaultdict

from .text import BOS, EOS, UNK, to_sentences, to_tokens

SMOOTHINGS = ("addk", "witten-bell")


def build(token_sents, n):
    """context (n-1 words) -> Counter of the words that follow it."""
    foll = defaultdict(Counter)
    for toks in token_sents:
        toks = [BOS] * (n - 1) + toks + [EOS]
        for i in range(len(toks) - n + 1):
            foll[tuple(toks[i : i + n - 1])][toks[i + n - 1]] += 1
    return foll


class NgramLM:
    """A smoothed n-gram language model: fit -> prob / suggest_next.

    n          order of the model (1 = unigram, 2 = bigram, 3 = trigram, ...)
    k          add-k constant (also used for the unigram base of Witten-Bell)
    smoothing  "addk"        P = (count + k) / (total + k*|V|) at order n only
               "witten-bell" interpolate order n with n-1, n-2, ... 1, so an
                             unseen context falls back to shorter ones
    min_count  words seen fewer times than this in training become <unk>, so
               the model has probability mass for words it has never met
    """

    def __init__(self, n=2, k=0.5, smoothing="addk", min_count=1):
        if smoothing not in SMOOTHINGS:
            raise ValueError(f"smoothing must be one of {SMOOTHINGS}")
        self.n, self.k, self.smoothing, self.min_count = n, k, smoothing, min_count

    # ---- training -------------------------------------------------------
    def fit(self, text):
        self.sents = to_sentences(text)
        tokenized = [to_tokens(s) for s in self.sents]
        word_counts = Counter(w for toks in tokenized for w in toks)
        self.vocab = {w for w, c in word_counts.items() if c >= self.min_count} | {EOS, UNK}
        mapped = [[self._map(w) for w in toks] for toks in tokenized]
        # counts[m][context of m-1 words] -> Counter(next word), for every order m <= n
        self.counts = {m: build(mapped, m) for m in range(1, self.n + 1)}
        self.totals = {m: {c: sum(f.values()) for c, f in d.items()} for m, d in self.counts.items()}
        return self

    def _map(self, w):
        return w if w in self.vocab else UNK

    def _context(self, text):
        """Last n-1 tokens of `text` (padded with <s>), unknown words -> <unk>."""
        toks = [self._map(w) for w in to_tokens(text)]
        padded = [BOS] * (self.n - 1) + toks
        return tuple(padded[len(padded) - (self.n - 1) :])

    # ---- probabilities --------------------------------------------------
    def prob(self, ctx, w):
        """Smoothed P(w | ctx). `ctx` is the previous n-1 tokens (already <unk>-mapped)."""
        ctx = tuple(ctx)[len(ctx) - (self.n - 1) :] if self.n > 1 else ()
        if self.smoothing == "addk":
            return self._addk(ctx, w)
        return self._witten_bell(ctx, w)

    def _addk(self, ctx, w):
        c = self.counts[self.n].get(ctx)
        total = self.totals[self.n].get(ctx, 0)
        count = c[w] if c else 0
        return (count + self.k) / (total + self.k * len(self.vocab))

    def _witten_bell(self, ctx, w):
        m = len(ctx) + 1
        if m == 1:  # unigram base case, add-k so every word keeps some mass
            c = self.counts[1][()]
            return (c[w] + self.k) / (self.totals[1][()] + self.k * len(self.vocab))
        lower = self._witten_bell(ctx[1:], w)
        c = self.counts[m].get(ctx)
        if not c:  # never saw this context -> trust the shorter one entirely
            return lower
        total, types = self.totals[m][ctx], len(c)
        return (c[w] + types * lower) / (total + types)

    # ---- next-word suggestions -----------------------------------------
    def suggest_next(self, text, topk=3):
        """Most likely next words, backing off to shorter contexts to fill topk.
        Never suggests end-of-sentence or <unk>."""
        full = self._context(text)
        out = []
        for m in range(self.n, 0, -1):
            ctx = full[len(full) - (m - 1) :] if m > 1 else ()
            c = self.counts[m].get(ctx, Counter())
            for w, _ in c.most_common():
                if w not in (EOS, UNK) and w not in out:
                    out.append(w)
                if len(out) == topk:
                    return out
        return out
