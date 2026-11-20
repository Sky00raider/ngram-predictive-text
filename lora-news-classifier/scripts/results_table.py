"""Collect every finished run into one table: outputs/results.md.

Reads outputs/runs/<run>/metrics.json (+ config.json) for every run except
smoke tests and the short Stage 4 reload check, one row per run.
Run from the project root:  python scripts/results_table.py
"""
import json
from pathlib import Path

RUNS_DIR = Path("outputs/runs")
OUT_PATH = Path("outputs/results.md")
SKIP_PREFIXES = ("smoke", "reload_check")


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if d.name.startswith(SKIP_PREFIXES) or not (d / "metrics.json").exists():
            continue
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        runs.append({"name": d.name, "m": metrics, "c": config})
    return runs


def main():
    runs = load_runs()
    labels = runs[0]["c"]["labels"]
    lines = ["# Results", "",
             "Test set: 630 headlines; \"unseen\" = the 524 test headlines that do not appear in train.", "",
             "| run | base | targets | rank | quant | trainable params | adapter | val acc | test acc | "
             "test acc (unseen) | " + " | ".join(f"{l} (test)" for l in labels) + " | s/step (CPU) | train time |",
             "|" + "---|" * (12 + len(labels))]
    for r in runs:
        m, c = r["m"], r["c"]
        per_class = " | ".join(f"{m['test']['per_class'][l]:.3f}" for l in labels)
        lines.append(
            f"| {r['name']} | {c['base']} | {c['targets']} | {c['rank']} | {c.get('quant') or '-'} | "
            f"{m['trainable_params']:,} | {m.get('adapter_bytes', 0) / 1024:,.0f} KB | "
            f"{m['validation']['acc']:.3f} | **{m['test']['acc']:.3f}** | {m['test']['acc_unseen']:.3f} | "
            f"{per_class} | {m['seconds_per_step']:.1f} | {m['seconds'] / 60:.0f} min |")
    report = "\n".join(lines) + "\n"
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
