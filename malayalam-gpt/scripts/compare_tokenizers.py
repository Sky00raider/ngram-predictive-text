"""Compare char-level, our BPE and GPT-2 on held-out (validation) Malayalam text.

Run from the project root (after scripts/train_bpe.py):  python scripts/compare_tokenizers.py
"""
import random
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bpe import BPETokenizer
from src.data import load_corpus
from src.tokenizer import CharTokenizer

BPE_PATH = Path("outputs/bpe_vocab.json")
OUT_PATH = Path("outputs/tokenizer_comparison.md")
N_SENTENCES = 20
N_EXAMPLES = 3


def pick_sentences(val_text: str) -> list[str]:
    """Split paragraphs into sentences, keep 40-200 chars, take a seeded sample."""
    sentences = [s for para in val_text.split("\n") for s in para.split(". ") if 40 <= len(s) <= 200]
    return random.Random(0).sample(sentences, N_SENTENCES)


def main():
    text = load_corpus()
    val_text = text[int(0.9 * len(text)) :]  # last 10%, same cut as train_val_split
    sentences = pick_sentences(val_text)

    # Each tokenizer is just a function: str -> list of token ids.
    char_tok = CharTokenizer.from_text(text)
    bpe_tok = BPETokenizer.load(BPE_PATH)
    gpt2 = tiktoken.get_encoding("gpt2")
    tokenizers = {
        f"char-level ({char_tok.vocab_size})": char_tok.encode,
        f"our BPE ({bpe_tok.vocab_size})": bpe_tok.encode,
        "GPT-2 (50257)": gpt2.encode,
    }

    rows = []
    for name, encode in tokenizers.items():
        per_sentence = [len(encode(s)) for s in sentences]
        mean_tokens = sum(per_sentence) / len(per_sentence)
        mean_chars = sum(len(s) for s in sentences) / len(sentences)
        total = len(encode(val_text))
        rows.append((name, mean_tokens, mean_chars / mean_tokens, total, len(val_text) / total))

    lines = [
        "# Tokenizer comparison",
        "",
        f"Sample: {N_SENTENCES} validation sentences (40-200 chars, seeded). "
        f"Whole validation text: {len(val_text):,} characters.",
        "",
        "| tokenizer | tokens / sentence | chars / token (sentences) | validation tokens | chars / token (validation) |",
        "|---|---|---|---|---|",
    ]
    for name, mt, cpt, total, vcpt in rows:
        lines.append(f"| {name} | {mt:.1f} | {cpt:.2f} | {total:,} | {vcpt:.2f} |")

    lines += ["", f"## {N_EXAMPLES} example sentences", ""]
    for s in sentences[:N_EXAMPLES]:
        lines.append(f"**{s}**  ({len(s)} chars)")
        lines.append("")
        for name, encode in [("char-level", char_tok.encode), ("our BPE", bpe_tok.encode)]:
            ids = encode(s)
            decoder = char_tok if name == "char-level" else bpe_tok
            lines.append(f"- {name} ({len(ids)} tokens): `{'|'.join(decoder.decode([i]) for i in ids)}`")
        lines.append(f"- GPT-2 ({len(gpt2.encode(s))} tokens): not shown, see note below")
        lines.append("")
    lines += [
        "Note: GPT-2 tokens are byte fragments. A Malayalam character is 3 bytes in UTF-8 and "
        "GPT-2 has almost no Malayalam merges, so a single character is often split across "
        "several tokens (and those fragments cannot be shown as readable text).",
    ]

    report = "\n".join(lines)
    print(report)
    OUT_PATH.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
