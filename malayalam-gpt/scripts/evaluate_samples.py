"""Measure generated text without needing to read it.

Two checks, for each model and a few decoding settings:
  * script errors: a dependent sign (vowel sign, virama, anusvara, visarga) that
    starts a word or follows another vowel sign/virama, which no real word has
  * real words: share of generated words that occur in the training corpus

Writes outputs/sample_quality.md
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_corpus
from src.generate import MiniGPT

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
PROMPTS = ["കേരളത്തിൽ", "ഇന്ന്", "സംസ്ഥാനത്ത്", "പൊലീസ്", "സിനിമ"]
CHARS_PER_SETTING = 3000

VOWEL_SIGNS = set(chr(c) for c in range(0x0D3E, 0x0D4D)) | {"ൗ"}
DEPENDENT = VOWEL_SIGNS | {"്", "ം", "ഃ"}  # + virama, anusvara, visarga
WORD = re.compile(r"[ഀ-ൿ]+")


def script_errors(text: str) -> int:
    errors = 0
    for word in WORD.findall(text):
        if word[0] in DEPENDENT:
            errors += 1  # word starts with a sign that must follow a consonant
        errors += sum(1 for a, b in zip(word, word[1:]) if a in VOWEL_SIGNS | {"്"} and b in VOWEL_SIGNS)
    return errors


def generate(gpt: MiniGPT, **kw) -> str:
    out, seed = [], 0
    while sum(map(len, out)) < CHARS_PER_SETTING:
        prompt = PROMPTS[seed % len(PROMPTS)]
        out.append(gpt.complete(prompt, 200, seed=seed, **kw)[len(prompt):])
        seed += 1
    return "\n".join(out)


def main():
    text = load_corpus()
    train_words = set(WORD.findall(text[: int(0.9 * len(text))]))
    val_words = WORD.findall(text[int(0.9 * len(text)):])
    random.seed(0)
    real = random.sample(val_words, 2000)

    rows = [("real validation text", "", sum(w in train_words for w in real) / len(real),
             script_errors(" ".join(real)) / len(real) * 100)]
    settings = [("temp 0.8, top-k 40", dict(temperature=0.8, top_k=40)),
                ("temp 0.5", dict(temperature=0.5)),
                ("temp 1.0", dict(temperature=1.0)),
                ("temp 1.4", dict(temperature=1.4))]
    for run in ["char", "bpe"]:
        gpt = MiniGPT(OUTPUTS / run)
        for name, kw in settings:
            words = WORD.findall(sample := generate(gpt, **kw))
            rows.append((run, name, sum(w in train_words for w in words) / len(words),
                         script_errors(sample) / len(words) * 100))
            print(rows[-1], flush=True)

    lines = ["# Sample quality (measured, no reading needed)\n",
             f"~{CHARS_PER_SETTING} generated characters per row, prompts {', '.join(PROMPTS)}.",
             "*Real words* = share of generated words that appear in the training text. "
             "*Script errors* = impossible sign sequences per 100 words.\n",
             "| model | decoding | real words | script errors / 100 words |", "|---|---|---|---|"]
    lines += [f"| {m} | {d} | {r:.0%} | {e:.1f} |" for m, d, r, e in rows]
    (OUTPUTS / "sample_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
