"""Text -> sentences -> tokens."""
import re

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def to_sentences(text: str) -> list[str]:
    """Split on ., ! and ? (naive: "Mr." also splits, which is fine at this scale)."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def to_tokens(sentence: str) -> list[str]:
    """Lower-case words; apostrophes are kept so "don't" stays one token.
    Curly quotes (common in Gutenberg texts) are normalised to a plain apostrophe."""
    sentence = sentence.lower().replace("’", "'").replace("‘", "'")
    return [w.strip("'") for w in re.findall(r"[a-z']+", sentence) if w.strip("'")]
