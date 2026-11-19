"""Text cleaning for the Malayalam corpus.

`clean_text` is a pure function: raw text in, cleaned text out
(one paragraph per line). Small helpers below do one job each.
"""
import re
import unicodedata

VIRAMA = "\u0d4d"
ZWJ, ZWNJ = "\u200d", "\u200c"

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

# Character-level replacements: Malayalam digits, curly quotes, dashes.
CHAR_MAP = {chr(0x0D66 + i): str(i) for i in range(10)}
CHAR_MAP.update({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                 "\u2013": "-", "\u2014": "-"})
CHAR_MAP = str.maketrans(CHAR_MAP)

PUNCTUATION = set(".,?!'\"()-:;")
KEEP_ASCII = set("0123456789 ") | PUNCTUATION

MIN_LINE_CHARS = 20
MIN_MALAYALAM_RATIO = 0.70


def is_malayalam(ch: str) -> bool:
    """A Malayalam letter or sign (digits excluded), unassigned code points excluded."""
    in_block = "\u0d00" <= ch <= "\u0d7f"
    is_digit = "\u0d66" <= ch <= "\u0d6f"
    return in_block and not is_digit and unicodedata.category(ch) != "Cn"


def normalize(text: str) -> str:
    """NFC, legacy chillus -> atomic chillus, then drop ZWJ/ZWNJ."""
    text = unicodedata.normalize("NFC", text)
    text = CHILLU_RE.sub(lambda m: CHILLU[m.group(1)], text)
    return text.replace(ZWJ, "").replace(ZWNJ, "")


def malayalam_ratio(line: str) -> float:
    """Share of non-space characters that are Malayalam letters/signs."""
    chars = [c for c in line if not c.isspace()]
    if not chars:
        return 0.0
    return sum(is_malayalam(c) for c in chars) / len(chars)


def keep_allowed(line: str) -> str:
    """Map digits/quotes/dashes, then replace every disallowed char with a space."""
    line = line.translate(CHAR_MAP)
    return "".join(c if is_malayalam(c) or c in KEEP_ASCII else " " for c in line)


def collapse_spaces(line: str) -> str:
    return " ".join(line.split())


def clean_line(line: str) -> str:
    """Clean one line; return '' if the line should be dropped."""
    if malayalam_ratio(line) < MIN_MALAYALAM_RATIO:  # judged before Latin is stripped
        return ""
    line = collapse_spaces(keep_allowed(line))
    return line if len(line) >= MIN_LINE_CHARS else ""


def clean_text(text: str) -> str:
    """Clean a whole text: one paragraph per line, no blanks, no duplicates."""
    seen, out = set(), []
    for line in normalize(text).replace("\r", "\n").split("\n"):
        line = clean_line(line)
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out) + "\n" if out else ""
