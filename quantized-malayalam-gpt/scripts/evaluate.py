"""Build one variant of the P2 model, benchmark it, and save the result.

    python scripts/evaluate.py --method fp32           # -> outputs/results/fp32.json
    python scripts/evaluate.py --method dynamic
    python scripts/evaluate.py --method static_real    # calibrated on real Malayalam windows
    python scripts/evaluate.py --method static_randn   # calibrated on torch.randn "embeddings"
    python scripts/evaluate.py --method qat            # ~5 min fine-tune; --reuse skips it if
                                                       # outputs/models/qat_prepared.pt exists

Everything runs on CPU: PyTorch's int8 kernels are CPU-only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src import quantize as Q  # noqa: E402
from src.benchmark import BenchConfig, benchmark  # noqa: E402
from src.p2 import count_params, load_fp32_model, load_splits, load_tokenizer  # noqa: E402

RESULTS = ROOT / "outputs" / "results"
QAT_CKPT = ROOT / "outputs" / "models" / "qat_prepared.pt"
METHODS = ["fp32", "dynamic", "static_real", "static_randn", "qat"]
TRAIN_THREADS = 8  # QAT fine-tuning uses every core, like P2's training; benchmarks use BenchConfig.num_threads


def build(method: str, train: torch.Tensor, reuse: bool) -> tuple:
    """Return (model, extra info for the result file)."""
    model, cfg, _ = load_fp32_model()
    if method == "fp32":
        return model, {}
    if method == "dynamic":
        return Q.dynamic_int8(model), {}
    if method == "static_real":
        m, ranges = Q.static_int8(model, Q.real_calibration(train, cfg.block_size))
        return m, {"calibration": "16 batches x 32 real training windows", "activation_ranges": ranges}
    if method == "static_randn":
        m, ranges = Q.static_int8(model, Q.randn_calibration(cfg.block_size, cfg.n_embd), from_embeddings=True)
        return m, {"calibration": "16 batches of torch.randn(32, 128, 128) fed in place of the embeddings",
                   "activation_ranges": ranges}
    if method == "qat":
        prepared = Q.prepare_qat_model(model)
        extra = {}
        if reuse and QAT_CKPT.exists():
            prepared.load_state_dict(torch.load(QAT_CKPT))
            extra = json.loads((QAT_CKPT.with_suffix(".json")).read_text(encoding="utf-8"))
            print(f"reused {QAT_CKPT.relative_to(ROOT)}")
        else:
            torch.set_num_threads(TRAIN_THREADS)
            torch.manual_seed(0)
            start = time.perf_counter()
            history = Q.qat_finetune(prepared, train, cfg.block_size)
            extra = {"qat_minutes": (time.perf_counter() - start) / 60, "qat_history": history,
                     "qat_settings": "300 steps, AdamW lr 1e-4, batch 32, observers frozen at step 200, "
                                     f"{TRAIN_THREADS} threads"}
            QAT_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prepared.state_dict(), QAT_CKPT)
            QAT_CKPT.with_suffix(".json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
            print(f"qat fine-tune took {extra['qat_minutes']:.1f} min")
        extra["activation_ranges"] = Q.activation_ranges(prepared)
        return Q.convert_qat(prepared), extra
    raise ValueError(f"unknown method {method!r}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which has no Malayalam
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="fp32", choices=METHODS)
    p.add_argument("--reuse", action="store_true", help="qat: load the saved fine-tuned model instead of training")
    a = p.parse_args()

    tok = load_tokenizer()
    train, val = load_splits(tok)
    _, cfg, info = load_fp32_model()
    model, extra = build(a.method, train, a.reuse)
    bc = BenchConfig()
    r = benchmark(a.method, model, tok, val, cfg.block_size, bc)
    r["params"] = count_params(model) if a.method == "fp32" else None
    r["p2_checkpoint"] = info
    r.update(extra)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{a.method}.json"
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
