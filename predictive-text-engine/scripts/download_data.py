"""Download public-domain books from Project Gutenberg into data/.

Usage:  python scripts/download_data.py

Each book is saved with the Gutenberg licence header/footer stripped, then all
books are concatenated into data/corpus.txt.
"""
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

BOOKS = {
    1342: "pride_and_prejudice",
    11: "alice_in_wonderland",
    1661: "adventures_of_sherlock_holmes",
    84: "frankenstein",
    345: "dracula",
    98: "a_tale_of_two_cities",
    1400: "great_expectations",
    158: "emma",
    1260: "jane_eyre",
    768: "wuthering_heights",
    2701: "moby_dick",
    76: "huckleberry_finn",
    74: "tom_sawyer",
    120: "treasure_island",
    161: "sense_and_sensibility",
    730: "oliver_twist",
    766: "david_copperfield",
    1257: "three_musketeers",
    43: "jekyll_and_hyde",
    174: "picture_of_dorian_gray",
}

DATA = Path(__file__).resolve().parent.parent / "data"


def strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    return text[start.end() if start else 0 : end.start() if end else len(text)]


def fetch(url: str) -> bytes:
    """GET a URL. Falls back to curl if Python can't verify certificates
    (common with a python.org install on Windows). Verification is never disabled."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ngram-lm-student-project"})
        return urllib.request.urlopen(req, timeout=60).read()
    except urllib.error.URLError:
        return subprocess.run(["curl", "-sSfL", url], check=True, capture_output=True).stdout


def main() -> None:
    DATA.mkdir(exist_ok=True)
    parts = []
    for book_id, name in BOOKS.items():
        path = DATA / f"{name}.txt"
        if not path.exists():
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            print(f"downloading {name} ({url})")
            raw = fetch(url).decode("utf-8-sig", "replace")
            path.write_text(strip_gutenberg(raw), encoding="utf-8")
        parts.append(path.read_text(encoding="utf-8"))
    corpus = "\n\n".join(parts)
    (DATA / "corpus.txt").write_text(corpus, encoding="utf-8")
    print(f"wrote data/corpus.txt: {len(corpus):,} characters, {len(corpus.split()):,} words")


if __name__ == "__main__":
    main()
