"""Download the iNLTK Malayalam news-headline dataset into data/.

Source: IndicGLUE, config `inltkh.ml` on the Hugging Face Hub
(ai4bharat/indic_glue). About 0.6 MB: 5,036 train / 630 validation / 630 test
headlines, each with a category label.

The dataset was published with a loading script, which recent versions of
`datasets` no longer run, so we read the Hub's auto-converted Parquet files.

Writes data/{train,validation,test}.jsonl, one {"text", "label"} per line,
with the label as its name (e.g. "sports"), not an integer.
Run from the project root:  python scripts/download_data.py
"""
import json
from pathlib import Path

from datasets import load_dataset

REPO = "ai4bharat/indic_glue"
CONFIG = "inltkh.ml"
OUT_DIR = Path("data")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    ds = load_dataset(REPO, data_dir=CONFIG, revision="refs/convert/parquet")
    label_names = ds["train"].features["label"].names  # shared across languages
    for split, rows in ds.items():
        path = OUT_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                record = {"text": row["text"], "label": label_names[row["label"]]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split:>10}: {len(rows):5d} rows -> {path}")


if __name__ == "__main__":
    main()
