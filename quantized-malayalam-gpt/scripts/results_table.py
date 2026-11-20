"""Collect outputs/results/*.json into the tables used by the README.

    python scripts/results_table.py
      -> outputs/results.md              one row per method
      -> outputs/samples_side_by_side.md same prompt, same seed, every variant
      -> outputs/calibration_ranges.md   static PTQ: real-text vs torch.randn activation ranges
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import METHODS, RESULTS  # noqa: E402

LABELS = {
    "fp32": "FP32 baseline (P2 checkpoint)",
    "dynamic": "Dynamic int8",
    "static_real": "Static int8, calibrated on real text",
    "static_randn": "Static int8, calibrated on torch.randn",
    "qat": "QAT int8 (300-step fake-quant fine-tune)",
    "fp32_finetuned": "Control: FP32, same 300-step fine-tune",
}
OUT = ROOT / "outputs"


def short(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    R = {m: json.loads((RESULTS / f"{m}.json").read_text(encoding="utf-8")) for m in METHODS}
    base = R["fp32"]

    lines = [
        "# Results\n",
        f"CPU only ({base['env']['cpu']}), torch {base['env']['torch']}, int8 engine {base['env']['quant_engine']}, "
        f"1 thread. Size = saved state dict. Generation = {base['bench']['gen_tokens']} new tokens from "
        f"`{base['bench']['prompt']}`, batch 1; forward = one 32x128 batch; both are medians of interleaved rounds "
        "(`outputs/speed.md`). Val loss = 40 fixed batches of 32x128 from P2's val split, the same windows for every row. "
        "Δ val loss is against the FP32 baseline; for QAT the fair reference is the control row.\n",
        "| method | size (MB) | Linear weights (MB) | generation tok/s | forward tok/s | val loss | Δ val loss | sample |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        r, s = R[m], R[m].get("speed")
        gen = f"{s['gen_tokens_per_sec']:.0f}" if s else f"{r['tokens_per_sec']:.0f}"
        fwd = f"{s['fwd_tokens_per_sec']:,.0f}" if s else f"{r['forward_tokens_per_sec']:,.0f}"
        delta = r["val_loss"] - base["val_loss"]
        if m == "qat":
            delta_s = f"{delta:+.4f} ({r['val_loss'] - R['fp32_finetuned']['val_loss']:+.4f} vs control)"
        else:
            delta_s = "" if m == "fp32" else f"{delta:+.4f}"
        lines.append(
            f"| {LABELS[m]} | {r['size_total_bytes'] / 1e6:.2f} | {r['size_linear_bytes'] / 1e6:.2f} | {gen} | {fwd} | "
            f"{r['val_loss']:.4f} | {delta_s} | {short(r['sample'])} |"
        )
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[3:]))

    side = [
        "# Samples side by side\n",
        f"Prompt `{base['bench']['prompt']}`, seed {base['bench']['seed']}, temperature {base['bench']['temperature']}, "
        f"top-k {base['bench']['top_k']}, {base['bench']['gen_tokens']} new characters. The same seed means the same "
        "random numbers, so variants agree until a quantized distribution first picks a different character.\n",
    ]
    ref = base["sample"]
    for m in METHODS:
        same = next((i for i, (a, b) in enumerate(zip(ref, R[m]["sample"])) if a != b), len(ref))
        note = "" if m == "fp32" else f" (identical to FP32 for the first {same} characters)"
        side.append(f"## {LABELS[m]}{note}\n\n{R[m]['sample'].strip()}\n\n_English translation: TODO_\n")
    (OUT / "samples_side_by_side.md").write_text("\n".join(side), encoding="utf-8")

    real, rnd = R["static_real"]["activation_ranges"], R["static_randn"]["activation_ranges"]
    cal = [
        "# Calibration: real text vs torch.randn\n",
        "(min, max) recorded by each Linear's input observer during static calibration. Only head 0 is shown; "
        "key/query/value share one input, so they see the same range. Width ratio = randn range / real range: "
        "below 1 means randn calibration would clip real activations.\n",
        "| layer | input comes from | real | randn | width ratio |",
        "|---|---|---|---|---|",
    ]
    source = {"query": "LayerNorm ln1", "proj": "concatenated attention heads", "net.0": "LayerNorm ln2",
              "net.2": "ReLU", "lm_head": "final LayerNorm ln_f"}
    for k in real:
        if re.search(r"heads\.[1-3]\.|\.(key|value)$", k):
            continue
        (a, b), (c, d) = real[k], rnd[k]
        src = next(v for s, v in source.items() if k.endswith(s))
        cal.append(f"| `{k}` | {src} | [{a:.2f}, {b:.2f}] | [{c:.2f}, {d:.2f}] | {(d - c) / (b - a):.2f} |")
    (OUT / "calibration_ranges.md").write_text("\n".join(cal) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
