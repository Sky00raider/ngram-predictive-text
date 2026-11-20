"""Why the tokenizer matters for Malayalam: distilgpt2 vs bloom-560m.

For every headline (all splits, after cleaning) counts the tokens each base
model's tokenizer produces, then shows how three example headlines are split.
Writes outputs/tokenizer_comparison.md.
Run from the project root:  python scripts/compare_tokenizers.py
"""
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from src.config import BASE_MODELS
from src.data import load_all

OUT_PATH = Path("outputs/tokenizer_comparison.md")


def pieces(tok, text: str) -> list[str]:
    """Each token as the text it stands for, or '·' if it is only part of a character.

    GPT-2 works on UTF-8 bytes: a Malayalam letter is 3 bytes and often spans
    several tokens, none of which decodes to readable text on its own.
    """
    out = []
    for i in tok(text, add_special_tokens=False).input_ids:
        piece = tok.decode([i])
        out.append("·" if "�" in piece else piece)
    return out


def main():
    data = load_all()
    texts = [text for rows in data.values() for text, _ in rows]
    chars = [len(t) for t in texts]
    by_length = sorted(texts, key=len)
    examples = [by_length[int(q * (len(texts) - 1))] for q in (0.25, 0.5, 0.75)]  # short, typical, long

    lines = ["# Tokenizer comparison: distilgpt2 vs bloom-560m", "",
             f"{len(texts):,} headlines (all splits, cleaned), mean {mean(chars):.1f} characters.", "",
             "| tokenizer | vocab | mean tokens / headline | median | 95th pct | max | chars / token | "
             "tokens per character |",
             "|---|---|---|---|---|---|---|---|"]
    tokenizers = {}
    for base, spec in BASE_MODELS.items():
        tok = AutoTokenizer.from_pretrained(spec["hf_name"])
        tokenizers[base] = tok
        n = sorted(len(tok(t, add_special_tokens=False).input_ids) for t in texts)
        pct = lambda p: n[int(p * (len(n) - 1))]
        lines.append(f"| {base} | {len(tok):,} | {mean(n):.1f} | {pct(.5)} | {pct(.95)} | {n[-1]} | "
                     f"{sum(chars) / sum(n):.2f} | {sum(n) / sum(chars):.2f} |")

    lines += ["", "## Example headlines", "",
              "Tokens separated by `|`; `·` marks a token that is only a fragment of a "
              "character's UTF-8 bytes (not readable on its own).", ""]
    for text in examples:
        lines.append(f"**{text}** ({len(text)} chars)")
        lines.append("")
        for base, tok in tokenizers.items():
            p = pieces(tok, text)
            lines.append(f"- {base} ({len(p)} tokens): `{'|'.join(p)}`")
        lines.append("")

    report = "\n".join(lines) + "\n"
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
