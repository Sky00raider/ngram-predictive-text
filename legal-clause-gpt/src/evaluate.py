"""Evaluate a trained run from its saved logits; no GPU and no model weights needed.

    python -m src.evaluate --run gpt2_n60k [--model-dir path/to/model]

Reads outputs/runs/<run>/{val,test}_logits.npz, config.json and summary.json and writes,
into the same folder:
  report.md          every table below, ready to paste into the README
  metrics.json       the headline numbers (and the chosen review threshold)
  per_class.csv      support, train count, precision, recall, F1, review rate per class
  coverage.png       accuracy vs coverage as the review threshold rises
  rare_classes.png   per-class accuracy and review rate vs training frequency
  calibration.png    reliability diagram, before and after temperature scaling

Honesty rules: every threshold and the temperature are fitted on *validation*
and applied unchanged to *test*. Test is only ever scored.
With --model-dir, the chosen threshold is also written into the model's
tagger_config.json so LegalClauseTagger uses it by default.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import metrics as M  # noqa: E402
from src.data import load_labels  # noqa: E402

RUNS_DIR = Path("outputs/runs")
TARGETS = [0.90, 0.95, 0.98, 0.99]   # accuracy targets for the review threshold
DEFAULT_TARGET = 0.95                # the one LegalClauseTagger uses
SUFFIX_TOKENS = 5                    # "\nClause type:" in GPT-2 tokens (checked in tests/test_recipe.py)
BUCKETS = [(0, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, 10**9)]

# Plot style: reference palette, light surface, recessive axes and grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def load(run_dir: Path, split: str) -> dict:
    d = np.load(run_dir / f"{split}_logits.npz")
    logits, target = d["logits"], d["target"]
    probs = M.softmax(logits)
    pred = probs.argmax(axis=1)
    return {"logits": logits, "target": target, "probs": probs, "pred": pred,
            "conf": probs.max(axis=1), "correct": pred == target,
            "seen": d["seen"], "n_tokens": d["n_tokens"]}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def evaluate(run: str, model_dir: str | None = None) -> dict:
    run_dir = RUNS_DIR / run
    cfg = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    labels = load_labels()
    k = len(labels)
    train_counts = np.array(summary["train_counts"])
    val, test = load(run_dir, "val"), load(run_dir, "test")
    room = cfg["max_len"] - SUFFIX_TOKENS - 1
    lines: list[str] = []
    say = lines.append
    out: dict = {"run": run, "base": cfg["base"], "train_clauses": summary["train_clauses"],
                 "max_len": cfg["max_len"], "train_minutes": round(summary["train_seconds"] / 60, 1)}

    # ---- headline -----------------------------------------------------------
    t = test
    acc, f1 = float(t["correct"].mean()), M.macro_f1(t["pred"], t["target"], k)
    unseen, trunc = ~t["seen"], t["n_tokens"] > room
    out.update(val_accuracy=float(val["correct"].mean()), val_macro_f1=M.macro_f1(val["pred"], val["target"], k),
               test_accuracy=acc, test_macro_f1=f1,
               test_accuracy_unseen=float(t["correct"][unseen].mean()),
               test_accuracy_truncated=float(t["correct"][trunc].mean()) if trunc.any() else None,
               test_accuracy_not_truncated=float(t["correct"][~trunc].mean()))
    say(f"# Evaluation: `{run}`\n")
    say(f"{cfg['base']}, full fine-tuning on {summary['train_clauses']:,} clauses, max_len {cfg['max_len']}, "
        f"trained in {out['train_minutes']} min on {summary['device']}.\n")
    say("## Headline (test split, 10,000 clauses, 100 classes)\n")
    say("| | accuracy | macro-F1 |")
    say("|---|---|---|")
    say(f"| validation | {pct(out['val_accuracy'])} | {pct(out['val_macro_f1'])} |")
    say(f"| **test** | **{pct(acc)}** | **{pct(f1)}** |")
    say(f"| test, clauses never seen verbatim in training ({unseen.sum():,}) | {pct(out['test_accuracy_unseen'])} | "
        f"{pct(M.macro_f1(t['pred'][unseen], t['target'][unseen], k))} |")
    say(f"| test, clause cut to fit max_len ({trunc.sum():,}) | "
        f"{pct(out['test_accuracy_truncated']) if trunc.any() else '-'} | |")
    say(f"| test, clause fits uncut ({(~trunc).sum():,}) | {pct(out['test_accuracy_not_truncated'])} | |")
    say("\nMacro-F1 weighs all 100 clause types equally, so it is pulled down by the rare ones; "
        "accuracy is dominated by the common ones.\n")

    # ---- confusion ----------------------------------------------------------
    say("## Most confused pairs (test)\n")
    say("| # | clause type A | clause type B | A predicted as B | B predicted as A | total |")
    say("|---|---|---|---|---|---|")
    pairs = M.confused_pairs(t["pred"], t["target"], top=15)
    for i, (a, b, ab, ba) in enumerate(pairs, 1):
        say(f"| {i} | {labels[a]} | {labels[b]} | {ab} | {ba} | {ab + ba} |")
    n_err = int((~t["correct"]).sum())
    share = sum(ab + ba for *_, ab, ba in pairs) / max(n_err, 1)
    say(f"\nThese 15 pairs account for {pct(share)} of all {n_err:,} test errors.\n")
    out["confused_pairs"] = [{"a": labels[a], "b": labels[b], "a_as_b": ab, "b_as_a": ba} for a, b, ab, ba in pairs]

    # ---- abstention ---------------------------------------------------------
    say("## Abstention: \"needs human review\" below a confidence threshold\n")
    say("Thresholds are chosen on validation as the lowest one that reaches the target accuracy, "
        "then applied unchanged to test.\n")
    say("| target accuracy | threshold (from val) | val coverage | val accuracy | **test coverage** | **test accuracy** "
        "| test clauses sent to review |")
    say("|---|---|---|---|---|---|---|")
    out["abstention"] = []
    for target_acc in TARGETS:
        thr = M.threshold_for_accuracy(val["conf"], val["correct"], target_acc)
        v, te = M.at_threshold(val["conf"], val["correct"], thr), M.at_threshold(t["conf"], t["correct"], thr)
        out["abstention"].append({"target": target_acc, **{f"test_{x}": y for x, y in te.items()},
                                  "val_coverage": v["coverage"], "val_accuracy": v["accuracy"]})
        say(f"| {pct(target_acc)} | {thr:.3f} | {pct(v['coverage'])} | {pct(v['accuracy'])} | "
            f"**{pct(te['coverage'])}** | **{pct(te['accuracy'])}** | {te['abstained']:,} |")
    say("\nFixed thresholds, for reference (test):\n")
    say("| threshold | coverage | accuracy |")
    say("|---|---|---|")
    for thr in (0.5, 0.7, 0.9):
        te = M.at_threshold(t["conf"], t["correct"], thr)
        say(f"| {thr} | {pct(te['coverage'])} | {pct(te['accuracy'])} |")
    if any(a["test_coverage"] == 0 for a in out["abstention"]):
        say("\nA coverage of 0% means validation never reached that accuracy at any threshold.")
    say("")
    chosen = next(a for a in out["abstention"] if a["target"] == DEFAULT_TARGET)
    out["review_threshold"] = chosen["test_threshold"]
    plot_coverage(t, out["abstention"], run_dir / "coverage.png", cfg["base"])

    # ---- rare classes -------------------------------------------------------
    stats = M.per_class(t["pred"], t["target"], k)
    review = np.array([np.mean(t["conf"][t["target"] == c] < out["review_threshold"]) if stats["support"][c] else
                       np.nan for c in range(k)])
    rho_acc = M.spearman(train_counts, stats["recall"])
    rho_review = M.spearman(train_counts, review)
    out.update(spearman_trainfreq_accuracy=rho_acc, spearman_trainfreq_review=rho_review)
    say("## Rare classes: per-class accuracy vs training frequency (test)\n")
    say(f"Review threshold here: {out['review_threshold']:.3f} (the 95%-accuracy threshold above).\n")
    say("| training examples per class | classes | mean per-class accuracy | mean F1 | share sent to review |")
    say("|---|---|---|---|---|")
    for lo, hi in BUCKETS:
        m = (train_counts >= lo) & (train_counts < hi)
        if m.any():
            rng = f"{lo}+" if hi > 10**8 else f"{lo}-{hi - 1}"
            say(f"| {rng} | {m.sum()} | {pct(stats['recall'][m].mean())} | {pct(stats['f1'][m].mean())} | "
                f"{pct(np.nanmean(review[m]))} |")
    say(f"\nSpearman rank correlation with training frequency: per-class accuracy {rho_acc:+.2f}, "
        f"review rate {rho_review:+.2f}.\n")
    say("Ten worst clause types (test):\n")
    say("| clause type | train examples | test clauses | accuracy | F1 | most often predicted as |")
    say("|---|---|---|---|---|---|")
    for c in np.argsort(stats["recall"])[:10]:
        wrong = t["pred"][(t["target"] == c) & ~t["correct"]]
        top = labels[np.bincount(wrong, minlength=k).argmax()] if len(wrong) else "-"
        say(f"| {labels[c]} | {train_counts[c]} | {stats['support'][c]} | {pct(stats['recall'][c])} | "
            f"{pct(stats['f1'][c])} | {top} |")
    say("\nClasses with fewer than 10 test clauses have very noisy per-class accuracy "
        f"({(stats['support'] < 10).sum()} such classes).\n")
    plot_rare(train_counts, stats, review, labels, run_dir / "rare_classes.png", cfg["base"])
    with (run_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "train_count", "test_support", "precision", "recall", "f1", "review_rate"])
        for c in range(k):
            w.writerow([labels[c], train_counts[c], stats["support"][c], f"{stats['precision'][c]:.4f}",
                        f"{stats['recall'][c]:.4f}", f"{stats['f1'][c]:.4f}", f"{review[c]:.4f}"])

    # ---- calibration --------------------------------------------------------
    temp = M.fit_temperature(val["logits"], val["target"])
    raw = M.reliability(t["conf"], t["correct"])
    scaled_conf = M.softmax(t["logits"], temp).max(axis=1)
    scaled = M.reliability(scaled_conf, t["correct"])
    out.update(ece=raw["ece"], temperature=temp, ece_after_temperature=scaled["ece"],
               nll=M.nll(t["logits"], t["target"]), nll_after_temperature=M.nll(t["logits"], t["target"], temp))
    say("## Calibration: are 90%-confident predictions right about 90% of the time? (test)\n")
    say("| confidence bin | clauses | mean confidence | accuracy | gap |")
    say("|---|---|---|---|---|")
    for i in range(len(raw["count"])):
        if raw["count"][i]:
            say(f"| {raw['edges'][i]:.1f}-{raw['edges'][i + 1]:.1f} | {raw['count'][i]:,} | "
                f"{pct(raw['confidence'][i])} | {pct(raw['accuracy'][i])} | "
                f"{100 * (raw['accuracy'][i] - raw['confidence'][i]):+.1f} pts |")
    gap = t["conf"].mean() - acc
    say(f"\nMean confidence {pct(t['conf'].mean())} vs accuracy {pct(acc)}: on average the model is "
        f"**{'over' if gap > 0 else 'under'}confident by {100 * abs(gap):.1f} points**. "
        f"ECE = {pct(raw['ece'])}.\n")
    say(f"Temperature scaling (one number T fitted on validation; logits / T): T = {temp:.2f}, "
        f"test ECE {pct(raw['ece'])} -> {pct(scaled['ece'])}, test NLL {out['nll']:.3f} -> "
        f"{out['nll_after_temperature']:.3f}. Accuracy and the ranking of confidences are unchanged, "
        f"so the coverage curve is identical; only the numbers the threshold is expressed in move.\n")
    plot_calibration(raw, scaled, temp, t["conf"], run_dir / "calibration.png", cfg["base"])

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=1))
    if model_dir:
        tc = Path(model_dir) / "tagger_config.json"
        conf = json.loads(tc.read_text()) if tc.exists() else {}
        conf.update(threshold=out["review_threshold"], threshold_target_accuracy=DEFAULT_TARGET)
        tc.write_text(json.dumps(conf, indent=2))
        print(f"review threshold {out['review_threshold']:.3f} -> {tc}")
    print("\n".join(lines))
    print(f"\n-> {run_dir}/report.md, metrics.json, per_class.csv, coverage.png, rare_classes.png, calibration.png")
    return out


# ---- plots -----------------------------------------------------------------

def plot_coverage(t: dict, abstention: list[dict], path: Path, base: str):
    cov, acc, _ = M.coverage_curve(t["conf"], t["correct"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(100 * cov, 100 * acc, color=BLUE)
    for a in abstention:
        if a["target"] in (0.90, 0.95, 0.98) and a["test_coverage"] >= 0.05:
            x, y = 100 * a["test_coverage"], 100 * a["test_accuracy"]
            ax.plot(x, y, "o", ms=8, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"threshold {a['test_threshold']:.2f}\n{y:.1f}% acc, {x:.0f}% answered",
                        (x, y), textcoords="offset points", xytext=(-12, -30), ha="right", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(max(50, 100 * acc[-1] - 8), 100.5)
    ax.set_xlabel("coverage: share of clauses the tagger answers (%)")
    ax.set_ylabel("accuracy on answered clauses (%)")
    ax.set_title(f"Accuracy vs coverage, {base}, test split")
    ax.text(100, 100 * acc[-1] - 1.2, f"answer everything: {100 * acc[-1]:.1f}%", ha="right", va="top",
            fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rare(train_counts, stats, review, labels, path: Path, base: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, y, title in ((axes[0], stats["recall"], "Per-class accuracy"),
                         (axes[1], review, "Share sent to human review")):
        ax.scatter(train_counts, 100 * y, s=36, color=BLUE, edgecolors=SURFACE, linewidths=1, zorder=3)
        # median per bucket, as a step line
        xs, ys = [], []
        for lo, hi in BUCKETS:
            m = (train_counts >= lo) & (train_counts < hi)
            if m.any():
                xs.append(np.exp(np.log(train_counts[m]).mean()))
                ys.append(100 * np.nanmedian(y[m]))
        ax.plot(xs, ys, color=INK2, linewidth=1.5, linestyle="--", zorder=2, label="median per frequency bucket")
        ax.set_xscale("log")
        ax.set_xlabel("training examples of the class (log scale)")
        ax.set_ylabel("%")
        ax.set_title(f"{title} vs training frequency")
        ax.set_ylim(-3, 103)
    # name the four worst classes, staggering the labels so neighbours don't collide
    for c, dy in zip(np.argsort(stats["recall"])[:4], (-12, 8, -24, 20)):
        axes[0].annotate(labels[c], (train_counts[c], 100 * stats["recall"][c]), textcoords="offset points",
                         xytext=(8, dy), fontsize=7.5, color=INK2,
                         arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", fontsize=8)
    fig.suptitle(f"{base}, test split, one dot per clause type", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(raw, scaled, temp, conf, path: Path, base: str):
    fig, (ax, hx) = plt.subplots(2, 1, figsize=(6, 6.4), height_ratios=[3, 1.2], sharex=True)
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle=":", label="perfect calibration")
    for r, color, name in ((raw, BLUE, f"raw softmax (ECE {100 * raw['ece']:.1f}%)"),
                           (scaled, ORANGE, f"temperature T={temp:.2f} (ECE {100 * scaled['ece']:.1f}%)")):
        m = r["count"] > 0
        ax.plot(100 * r["confidence"][m], 100 * r["accuracy"][m], "-o", ms=6, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.set_ylabel("accuracy in bin (%)")
    ax.set_title(f"Reliability diagram, {base}, test split")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, 102)
    bins = np.linspace(0, 100, 21)
    hx.hist(100 * conf, bins=bins, color=BLUE, rwidth=0.85)
    hx.set_yscale("log")
    hx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    hx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    hx.set_ylabel("clauses (log)")
    hx.set_xlabel("confidence of the predicted clause type (%)")
    hx.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="folder name under outputs/runs/")
    p.add_argument("--model-dir", default=None, help="also store the chosen threshold in this model's tagger_config.json")
    a = p.parse_args()
    evaluate(a.run, a.model_dir)


if __name__ == "__main__":
    main()
