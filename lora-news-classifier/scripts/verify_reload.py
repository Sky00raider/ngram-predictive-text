"""Reload a saved adapter onto a fresh pretrained base and check it matches.

Compares against what the training run recorded on the same test headlines:
  * the label logits (should agree to float rounding),
  * every prediction (should be identical),
  * the accuracy (should be identical).
Also checks that a fresh base WITHOUT the adapter does not match, so the test
can actually fail. Appends the result to outputs/reload_check.md.

Run from the project root:  python scripts/verify_reload.py outputs/runs/<run>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.adapter import build_classifier, load_adapter
from src.config import Config, get_device

OUT_PATH = Path("outputs/reload_check.md")


def label_logits(clf, texts, batch_size=64):
    return torch.cat([clf.label_logits(texts[i:i + batch_size]).cpu() for i in range(0, len(texts), batch_size)])


def accuracy(logits, labels, label_names):
    gold = torch.tensor([label_names.index(l) for l in labels])
    return (logits.argmax(-1) == gold).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    run_dir = p.parse_args().run_dir
    device = get_device()

    recorded = torch.load(run_dir / "test_logits.pt", weights_only=True)
    texts, labels, before = recorded["texts"], recorded["labels"], recorded["logits"]

    clf, cfg = load_adapter(run_dir / "adapter.pt", device)
    after = label_logits(clf, texts)

    # Control: same base, fresh untrained LoRA + label rows.
    untrained = build_classifier(Config(**cfg.to_dict()), device).eval()
    control = label_logits(untrained, texts)

    diff = (after - before).abs().max().item()
    same_preds = torch.equal(after.argmax(-1), before.argmax(-1))
    acc_before, acc_after = accuracy(before, labels, cfg.labels), accuracy(after, labels, cfg.labels)
    acc_control = accuracy(control, labels, cfg.labels)
    size_kb = (run_dir / "adapter.pt").stat().st_size / 1024
    ok = diff < 1e-4 and same_preds and acc_before == acc_after

    lines = [
        f"## {run_dir.name}",
        "",
        f"adapter file: {size_kb:,.0f} KB (base model {cfg.hf_name} not included) | {len(texts)} test headlines",
        "",
        "| | test accuracy |",
        "|---|---|",
        f"| at the end of training | {acc_before:.4f} |",
        f"| fresh base + reloaded adapter | {acc_after:.4f} |",
        f"| fresh base, untrained adapter (control) | {acc_control:.4f} |",
        "",
        f"max |logit difference| after reload: {diff:.2e} | identical predictions: {same_preds} | "
        f"**{'MATCH' if ok else 'MISMATCH'}**",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    with OUT_PATH.open("a", encoding="utf-8") as f:
        f.write(report + "\n")
    if not ok:
        raise SystemExit("reloaded adapter does not match")


if __name__ == "__main__":
    main()
