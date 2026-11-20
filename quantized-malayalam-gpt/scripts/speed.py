"""Fair speed comparison: all six variants in one process, measured in interleaved rounds.

A laptop CPU's clock drifts with temperature and power state, so separate runs
of the same FP32 model measured 105-163 tokens/sec. Here every round times each
variant once (in a rotated order) and the median over rounds is reported, so
drift hits all variants alike.

    python scripts/speed.py      # needs outputs/models/*.pt from evaluate.py --method qat / fp32_finetuned
                                 # -> adds "speed" to each outputs/results/<method>.json, writes outputs/speed.md
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from evaluate import METHODS, RESULTS, build  # noqa: E402
from src.benchmark import BenchConfig, forward_tokens_per_sec, tokens_per_sec, val_windows  # noqa: E402
from src.p2 import load_fp32_model, load_splits, load_tokenizer  # noqa: E402

ROUNDS = 7


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tok = load_tokenizer()
    train, val = load_splits(tok)
    _, cfg, _ = load_fp32_model()
    models = {m: build(m, train, reuse=True)[0].eval() for m in METHODS}

    bc = BenchConfig(speed_runs=3)
    torch.set_num_threads(bc.num_threads)
    prompt = torch.tensor([tok.encode(bc.prompt)], dtype=torch.long)
    batch = next(val_windows(val, cfg.block_size, bc))[0]

    gen = {m: [] for m in METHODS}
    fwd = {m: [] for m in METHODS}
    for r in range(ROUNDS):
        order = METHODS[r % len(METHODS):] + METHODS[: r % len(METHODS)]
        for m in order:
            gen[m].append(tokens_per_sec(models[m], prompt, bc)["tokens_per_sec"])
            fwd[m].append(forward_tokens_per_sec(models[m], batch, bc))
        print(f"round {r + 1}/{ROUNDS}: " + ", ".join(f"{m} {gen[m][-1]:.0f}" for m in METHODS))

    lines = [
        "# Speed, interleaved\n",
        f"{ROUNDS} rounds; each round times every variant once (rotated order): {bc.speed_runs} generations of "
        f"{bc.gen_tokens} tokens from `{bc.prompt}`, and {bc.speed_runs} forward passes over one "
        f"{bc.batch_size}x{cfg.block_size} batch. {bc.num_threads} thread, engine {torch.backends.quantized.engine}. "
        "Median over rounds, with min-max.\n",
        "| method | generation tokens/sec | min-max | forward tokens/sec (batch) | min-max |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        speed = {
            "gen_tokens_per_sec": statistics.median(gen[m]), "gen_min": min(gen[m]), "gen_max": max(gen[m]),
            "fwd_tokens_per_sec": statistics.median(fwd[m]), "fwd_min": min(fwd[m]), "fwd_max": max(fwd[m]),
            "rounds": ROUNDS,
        }
        path = RESULTS / f"{m}.json"
        res = json.loads(path.read_text(encoding="utf-8"))
        res["speed"] = speed
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        lines.append(f"| {m} | {speed['gen_tokens_per_sec']:.1f} | {speed['gen_min']:.0f}-{speed['gen_max']:.0f} | "
                     f"{speed['fwd_tokens_per_sec']:,.0f} | {speed['fwd_min']:,.0f}-{speed['fwd_max']:,.0f} |")
    (ROOT / "outputs" / "speed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
