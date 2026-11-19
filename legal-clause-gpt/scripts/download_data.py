"""Download LEDGAR (LexGLUE) into data/.

Source: config `ledgar` of coastalcph/lex_glue on the Hugging Face Hub, about
27 MB of Parquet: 60,000 train / 10,000 validation / 10,000 test clauses from
SEC-filed contracts, each labelled with one of 100 clause types. The label is
derived from the clause's heading in the original contract (e.g.
"Governing Laws"), and the heading itself is not part of the text.

Writes data/{train,validation,test}.jsonl, one {"text", "label"} per line with
the label as its name, and data/labels.json with the 100 names in the
dataset's own order (that order defines the class indices everywhere else).
Run from the project root:  python scripts/download_data.py
"""
import json
from pathlib import Path

from datasets import load_dataset

REPO = "coastalcph/lex_glue"
CONFIG = "ledgar"
OUT_DIR = Path("data")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    ds = load_dataset(REPO, CONFIG)
    names = ds["train"].features["label"].names
    (OUT_DIR / "labels.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    print(f"{len(names)} labels -> {OUT_DIR / 'labels.json'}")
    for split, rows in ds.items():
        path = OUT_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps({"text": row["text"], "label": names[row["label"]]}) + "\n")
        print(f"{split:>10}: {len(rows):6d} rows -> {path}")


if __name__ == "__main__":
    main()
