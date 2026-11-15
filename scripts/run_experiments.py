"""Reproduce every number in the README.

Usage:  python scripts/download_data.py && python scripts/run_experiments.py

Writes results/results.md, results/samples.txt and (if matplotlib is
installed) results/learning_curve.png.
"""
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ngram_lm import NgramLM, to_sentences, to_tokens  # noqa: E402
from ngram_lm.data import split_text  # noqa: E402

KS = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
ORDERS = [1, 2, 3]
FRACTIONS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
KS_CURVE = [0.5, 0.05, 0.005]  # smaller grid for the (slow) learning curve
MIN_COUNT = 2  # words seen once in training become <unk>


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    corpus = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    train, dev, test = split_text(corpus, seed=0)
    n_tok = lambda t: sum(len(to_tokens(s)) for s in to_sentences(t))  # noqa: E731
    md = ["# Results\n", f"Corpus: {n_tok(corpus):,} word tokens in {len(to_sentences(corpus)):,} sentences "
          f"(Project Gutenberg novels). Split by sentence 80/10/10 (seed 0): "
          f"train {n_tok(train):,}, dev {n_tok(dev):,}, test {n_tok(test):,} tokens. "
          f"Words seen <{MIN_COUNT}x in training are mapped to `<unk>`.\n"]

    # 1. Perplexity on dev: n x k (add-k), plus Witten-Bell -------------------
    print("grid: n x k on dev ...")
    grid, best = {}, {}
    for n in ORDERS:
        for k in KS:
            grid[n, k] = NgramLM(n, k, "addk", MIN_COUNT).fit(train).perplexity(dev)
        best[n] = min(KS, key=lambda k: grid[n, k])
    wb = {n: NgramLM(n, 0.01, "witten-bell", MIN_COUNT).fit(train).perplexity(dev) for n in ORDERS}
    rows = [[f"k={k}"] + [f"{grid[n, k]:,.1f}" for n in ORDERS] for k in KS]
    rows.append(["**Witten-Bell** (back-off)"] + [f"{wb[n]:,.1f}" for n in ORDERS])
    md += ["## 1. Dev perplexity: order n vs smoothing k\n",
           "Lower is better. Unigram / bigram / trigram:\n",
           table(["smoothing", "n=1", "n=2", "n=3"], rows), ""]

    # 2. Final held-out TEST numbers with dev-chosen settings ----------------
    print("test ...")
    final_rows = []
    for n in ORDERS:
        a = NgramLM(n, best[n], "addk", MIN_COUNT).fit(train)
        w = NgramLM(n, 0.01, "witten-bell", MIN_COUNT).fit(train)
        final_rows.append([n, f"add-k (k={best[n]})", f"{a.perplexity(test):,.1f}"])
        final_rows.append([n, "Witten-Bell", f"{w.perplexity(test):,.1f}"])
    md += ["## 2. Test perplexity (k chosen on dev, test touched once)\n",
           table(["n", "smoothing", "test perplexity"], final_rows), ""]

    # 3. Does more data fix the trigram? ---------------------------------------
    print("learning curve ...")
    full_vocab = {w for w, c in Counter(w for s in to_sentences(train) for w in to_tokens(s)).items()
                  if c >= MIN_COUNT}
    sents = to_sentences(train)
    random.Random(1).shuffle(sents)
    curve = {(n, sm): [] for n in (2, 3) for sm in ("addk", "witten-bell")}
    sizes = []
    for f in FRACTIONS:
        part = ". ".join(sents[: int(len(sents) * f)])
        sizes.append(n_tok(part))
        for n in (2, 3):
            # k is re-tuned on dev at each size, so add-k is never handicapped
            curve[n, "addk"].append(min(NgramLM(n, k, "addk").fit(part, full_vocab).perplexity(dev) for k in KS_CURVE))
            curve[n, "witten-bell"].append(NgramLM(n, 0.01, "witten-bell").fit(part, full_vocab).perplexity(dev))
    rows = [[f"{int(f * 100)}%", f"{s:,}"] + [f"{curve[n, sm][i]:,.1f}" for n in (2, 3) for sm in ("addk", "witten-bell")]
            for i, (f, s) in enumerate(zip(FRACTIONS, sizes))]
    md += ["## 3. Does more data fix the trigram?\n",
           "Dev perplexity as the training set grows (fixed vocabulary; k re-tuned per size for add-k).\n",
           table(["train fraction", "tokens", "bigram add-k", "bigram W-B", "trigram add-k", "trigram W-B"], rows), ""]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for (n, sm), ys in curve.items():
            ax.plot(sizes, ys, marker="o", ls="-" if sm == "witten-bell" else "--",
                    label=f"{'bi' if n == 2 else 'tri'}gram, {sm}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training tokens"); ax.set_ylabel("dev perplexity (lower is better)")
        ax.set_title("Does more data fix the trigram?"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ROOT / "results" / "learning_curve.png", dpi=150)
        md += ["![learning curve](learning_curve.png)\n"]
    except ImportError:
        print("matplotlib not installed - skipping plot")

    # 4. Samples ---------------------------------------------------------------
    print("samples ...")
    samples = []
    for n in (1, 2, 3):
        lm = NgramLM(n, best[n], "witten-bell", MIN_COUNT).fit(train)
        samples.append(f"--- n={n} ---")
        samples += [f"[seed {s}] {lm.generate(seed=s, max_len=30)}" for s in range(5)]
        samples.append("")
    (ROOT / "results" / "samples.txt").write_text("\n".join(samples), encoding="utf-8")

    lm = NgramLM(3, 0.01, "witten-bell", MIN_COUNT).fit(corpus)
    sug = [f"| `{p}` | {', '.join(lm.suggest_next(p))} |" for p in
           ["i do not", "it was a", "my dear", "in the middle of", "she looked at"]]
    md += ["## 4. Autocomplete (trigram, trained on the full corpus)\n", "| typed | suggestions |", "|---|---|", *sug, ""]

    (ROOT / "results" / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
