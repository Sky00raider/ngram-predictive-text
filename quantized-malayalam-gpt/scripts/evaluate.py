"""Build one variant of the P2 model, benchmark it, and save the result.

    python scripts/evaluate.py --method fp32     # -> outputs/results/fp32.json

Everything runs on CPU: PyTorch's int8 kernels are CPU-only.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.benchmark import BenchConfig, benchmark  # noqa: E402
from src.p2 import count_params, load_fp32_model, load_splits, load_tokenizer  # noqa: E402

RESULTS = ROOT / "outputs" / "results"


def build(method: str):
    model, cfg, info = load_fp32_model()
    if method == "fp32":
        return model, cfg, info
    raise ValueError(f"unknown method {method!r}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which has no Malayalam
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="fp32", choices=["fp32"])
    p.add_argument("--threads", type=int, default=BenchConfig.num_threads)
    a = p.parse_args()

    tok = load_tokenizer()
    _, val = load_splits(tok)
    model, cfg, info = build(a.method)
    bc = BenchConfig(num_threads=a.threads)
    r = benchmark(a.method, model, tok, val, cfg.block_size, bc)
    r["params"] = count_params(model) if a.method == "fp32" else None
    r["p2_checkpoint"] = info

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "" if a.threads == BenchConfig.num_threads else f"_t{a.threads}"
    out = RESULTS / f"{a.method}{suffix}.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"method        {r['method']}  (torch {r['env']['torch']}, engine {r['env']['quant_engine']}, {bc.num_threads} thread(s))")
    print(f"size          {r['size_total_bytes'] / 1e6:.3f} MB total, {r['size_linear_bytes'] / 1e6:.3f} MB in Linear layers")
    print(f"speed         {r['tokens_per_sec']:.1f} +- {r['tokens_per_sec_std']:.1f} tokens/sec "
          f"({bc.gen_tokens} tokens, {bc.speed_runs} runs after {bc.warmup_runs} warmup)")
    print(f"forward       {r['forward_tokens_per_sec']:,.0f} tokens/sec (one {bc.batch_size}x{cfg.block_size} batch)")
    print(f"val loss      {r['val_loss']:.4f}  ({bc.val_batches} fixed batches of {bc.batch_size}x{cfg.block_size}; "
          f"P2 logged {info['val_loss']:.4f} at step {info['step']})")
    print(f"sample        {r['sample']}")
    print(f"wrote         {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
