"""Plot train/val loss for every run in outputs/ -> outputs/loss_curve.png.

Losses from different tokenizers are per *token*, so they aren't comparable
directly. The right panel converts them to bits per character:
    bits/char = loss_per_token / ln(2) / chars_per_token
"""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
CHARS_PER_TOKEN = {"char": 1.0, "bpe": 2.63}  # bpe value from outputs/tokenizer_comparison.md (validation text)


def read_run(run_dir: Path):
    rows = list(csv.DictReader(open(run_dir / "loss.csv")))
    steps = [int(r["step"]) for r in rows]
    return steps, [float(r["train_loss"]) for r in rows], [float(r["val_loss"]) for r in rows]


def main():
    runs = [d for d in sorted(OUTPUTS.iterdir()) if (d / "loss.csv").exists()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        tok = json.loads((run / "config.json").read_text())["tokenizer"]
        steps, train, val = read_run(run)
        line, = ax1.plot(steps, train, label=f"{run.name} train")
        ax1.plot(steps, val, "--", color=line.get_color(), label=f"{run.name} val")
        bpc = [v / math.log(2) / CHARS_PER_TOKEN[tok] for v in val]
        ax2.plot(steps, bpc, color=line.get_color(), label=f"{run.name} val")
        print(f"{run.name}: final train {train[-1]:.3f}, val {val[-1]:.3f}, best val {min(val):.3f} "
              f"= {min(bpc):.3f} bits/char")
    ax1.set(title="Cross-entropy loss per token", xlabel="step", ylabel="loss (nats)")
    ax2.set(title="Validation loss in bits per character", xlabel="step", ylabel="bits / char")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "loss_curve.png", dpi=130)
    print("wrote", OUTPUTS / "loss_curve.png")


if __name__ == "__main__":
    main()
