"""How many parameters LoRA trains, for each base model, rank and target set.

Builds each model on PyTorch's "meta" device (shapes only, no memory, only
config.json is downloaded), injects LoRA and counts. Writes
outputs/lora_params.md.
Run from the project root:  python scripts/count_params.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from src.config import BASE_MODELS, Config
from src.lora import count_parameters, inject_lora

OUT_PATH = Path("outputs/lora_params.md")


def main():
    lines = ["# LoRA trainable parameters", "",
             "LoRA A/B only (the new label-token embedding rows add a few thousand more).", "",
             "| base | total params | targets | layers wrapped | r=4 | r=8 | r=16 |",
             "|---|---|---|---|---|---|---|"]
    for base in BASE_MODELS:
        for targets in ("attn", "attn+mlp"):
            cells = []
            for rank in (4, 8, 16):
                cfg = Config(base=base, targets=targets, rank=rank)
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg.hf_name))
                total = sum(p.numel() for p in model.parameters())
                wrapped = inject_lora(model, cfg.target_modules(), rank, cfg.alpha)
                trainable, _ = count_parameters(model)
                cells.append(f"{trainable:,} ({trainable / total:.2%})")
            lines.append(f"| {base} | {total / 1e6:.0f}M | {targets} | {len(wrapped)} | " + " | ".join(cells) + " |")
    report = "\n".join(lines) + "\n"
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
