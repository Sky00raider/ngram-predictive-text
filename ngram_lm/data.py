"""Train / dev / test splitting."""
import random

from .text import to_sentences


def split_text(text, fractions=(0.8, 0.1, 0.1), seed=0):
    """Shuffle sentences (fixed seed) and cut into train / dev / test strings.

    Splitting by sentence keeps every split a mix of all books. Dev is for
    choosing k; test is touched once at the end.
    """
    sents = to_sentences(text)
    random.Random(seed).shuffle(sents)
    a = int(len(sents) * fractions[0])
    b = a + int(len(sents) * fractions[1])
    return tuple(". ".join(part) for part in (sents[:a], sents[a:b], sents[b:]))
