"""Pack the project's code into dist/legal-clause-gpt.zip for uploading to Colab.

Only what training needs: src/, scripts/, tests/, requirements.txt (no data,
outputs, virtualenv or caches). Once the project is pushed to GitHub, the
notebook can `git clone` instead and this script isn't needed.
Run from the project root:  python scripts/make_colab_zip.py
"""
import zipfile
from pathlib import Path

INCLUDE = ["src", "scripts", "tests", "requirements.txt"]
OUT = Path("dist/legal-clause-gpt.zip")


def main():
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = Path(item)
            files = [path] if path.is_file() else sorted(path.rglob("*.py"))
            for f in files:
                z.write(f, Path("legal-clause-gpt") / f)
                print(f"  {f}")
    print(f"-> {OUT.resolve()}")


if __name__ == "__main__":
    main()
