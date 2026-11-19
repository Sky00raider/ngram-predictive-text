"""Compare evaluated runs side by side (e.g. distilgpt2 vs gpt2).

Run `python -m src.evaluate --run <run>` for each run first, then:
    python scripts/compare_runs.py gpt2_n60k distilgpt2_n60k

Writes outputs/comparison.md (table) and outputs/coverage_comparison.png
(accuracy vs coverage on test, one line per run).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics as M  # noqa: E402
from src.evaluate import BLUE, INK2, ORANGE, RUNS_DIR, SURFACE, load, plt  # noqa: E402  (plt is styled there)

PARAMS = {"gpt2": "124M", "distilgpt2": "82M"}   # + 100 x 768 new label-token rows each
COLORS = [BLUE, ORANGE]


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def main():
    runs = sys.argv[1:]
    if not 1 <= len(runs) <= 2:
        sys.exit("usage: python scripts/compare_runs.py RUN [RUN]")
    ms = [json.loads((RUNS_DIR / r / "metrics.json").read_text()) for r in runs]

    def row(name, f):
        return f"| {name} | " + " | ".join(f(m) for m in ms) + " |"

    def ab(m, target, key):
        return next(a for a in m["abstention"] if a["target"] == target)[key]

    lines = ["# distilgpt2 vs gpt2\n", "| | " + " | ".join(f"`{r}`" for r in runs) + " |",
             "|---|" + "---|" * len(runs),
             row("parameters", lambda m: PARAMS.get(m["base"], "?")),
             row("training time (T4)", lambda m: f"{m['train_minutes']} min"),
             row("test accuracy", lambda m: pct(m["test_accuracy"])),
             row("test macro-F1", lambda m: pct(m["test_macro_f1"])),
             row("test accuracy, unseen clauses", lambda m: pct(m["test_accuracy_unseen"])),
             row("coverage at 95% accuracy", lambda m: pct(ab(m, 0.95, "test_coverage"))),
             row("coverage at 98% accuracy", lambda m: pct(ab(m, 0.98, "test_coverage"))),
             row("ECE (raw softmax)", lambda m: pct(m["ece"])),
             row("ECE after temperature scaling", lambda m: f"{pct(m['ece_after_temperature'])} (T={m['temperature']:.2f})"),
             row("rank corr. train frequency vs per-class accuracy", lambda m: f"{m['spearman_trainfreq_accuracy']:+.2f}")]
    out = Path("outputs/comparison.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    lowest = 100
    for run, m, color in zip(runs, ms, COLORS):
        t = load(RUNS_DIR / run, "test")
        cov, acc, _ = M.coverage_curve(t["conf"], t["correct"])
        ax.plot(100 * cov, 100 * acc, color=color, label=f"{m['base']} ({pct(m['test_accuracy'])} at full coverage)")
        lowest = min(lowest, 100 * acc[-1])
        x, y = 100 * ab(m, 0.95, "test_coverage"), 100 * ab(m, 0.95, "test_accuracy")
        ax.plot(x, y, "o", ms=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(max(50, lowest - 8), 100.5)
    ax.set_xlabel("coverage: share of clauses the tagger answers (%)")
    ax.set_ylabel("accuracy on answered clauses (%)")
    ax.set_title("Accuracy vs coverage, test split")
    ax.legend(loc="lower left", fontsize=9)
    ax.text(1, 0.02, "dots: threshold chosen on validation for 95% accuracy", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig("outputs/coverage_comparison.png", dpi=150)
    print(f"\n-> {out}, outputs/coverage_comparison.png")


if __name__ == "__main__":
    main()
