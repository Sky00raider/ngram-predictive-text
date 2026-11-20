"""Extra: how much each Linear layer's weights move when rounded to int8.

    python scripts/layer_error.py    # -> outputs/layer_error.md

Error per layer = mean |W - dequant(W_int8)| / mean |W|, so 0.01 means the int8 weights
are off by 1% of a typical weight. Static real and static randn store identical
weights (calibration only sets activation ranges), so static appears once.
For QAT the reference is its own fine-tuned float weights, since those are what got rounded.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from evaluate import QAT_CKPT  # noqa: E402
from src import quantize as Q  # noqa: E402
from src.p2 import load_fp32_model, load_splits, load_tokenizer  # noqa: E402


def rel_error(w: torch.Tensor, w_q: torch.Tensor) -> float:
    return ((w - w_q).abs().mean() / w.abs().mean()).item()


def int8_weight(module) -> torch.Tensor:
    """Dequantized weight of a quantized Linear, or of the Linear inside a QuantLinear."""
    if isinstance(module, Q.QuantLinear):
        module = module.linear
    return module.weight().dequantize()


def group(name: str) -> str:
    """Average the 4 heads' key/query/value layers into one row per block."""
    return re.sub(r"heads\.\d+\.", "heads.*.", name)


def main():
    fp32, cfg, _ = load_fp32_model()
    train, _ = load_splits(load_tokenizer())
    dynamic = Q.dynamic_int8(fp32)
    static, _ = Q.static_int8(fp32, Q.real_calibration(train, cfg.block_size, n_batches=2))
    prepared = Q.prepare_qat_model(fp32)
    prepared.load_state_dict(torch.load(QAT_CKPT))
    qat = Q.convert_qat(prepared)

    rows = defaultdict(lambda: defaultdict(list))
    for name, m in fp32.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        w = m.weight.detach()
        w_qat_float = prepared.get_submodule(name).linear.weight.detach()
        r = rows[group(name)]
        r["dynamic (per-tensor)"].append(rel_error(w, int8_weight(dynamic.get_submodule(name))))
        r["static (per-channel)"].append(rel_error(w, int8_weight(static.get_submodule(name))))
        r["QAT (per-channel)"].append(rel_error(w_qat_float, int8_weight(qat.get_submodule(name))))
        r["QAT drift from FP32"].append(rel_error(w, w_qat_float))

    cols = ["dynamic (per-tensor)", "static (per-channel)", "QAT (per-channel)", "QAT drift from FP32"]
    lines = [
        "# Per-layer weight quantization error\n",
        "mean |W - dequant(W_int8)| / mean |W| per Linear layer; key/query/value rows average the 4 heads. "
        "The last column is not a rounding error: it is how far QAT's fine-tune moved the float weights.\n",
        "| layer | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    totals = defaultdict(list)
    for name, r in rows.items():
        cells = []
        for c in cols:
            v = sum(r[c]) / len(r[c])
            totals[c] += r[c]
            cells.append(f"{100 * v:.2f}%")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines.append("| **mean over all 61 Linears** | " +
                 " | ".join(f"**{100 * sum(totals[c]) / len(totals[c]):.2f}%**" for c in cols) + " |")
    out = ROOT / "outputs" / "layer_error.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
