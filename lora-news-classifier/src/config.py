"""Device detection, base-model presets and the run config.

One `Config` describes one fine-tuning run: which base model, which layers get
LoRA adapters, the rank, and the training settings. Every script builds its
Config from these defaults plus command-line overrides, and saves it next to
the adapter so a run can always be reproduced.
"""
from dataclasses import asdict, dataclass, field

import torch


def get_device() -> str:
    """Best available device: NVIDIA GPU, then Apple GPU, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# The same idea ("attention only" vs "attention + MLP") has different layer
# names in each architecture, so each base model lists its own.
#   distilgpt2 (GPT-2): c_attn = fused Q,K,V projection; c_fc / mlp.c_proj = MLP.
#     GPT-2 also has attn.c_proj (attention output), which we leave alone so the
#     "attention" group matches the course: the fused QKV layer only.
#   bloom-560m: query_key_value = fused Q,K,V; dense_h_to_4h / dense_4h_to_h = MLP.
#
# max_len (prompt tokens) is set per model so that at least 95% of headlines fit
# uncut. The same headline costs very different numbers of tokens:
#   distilgpt2: mean 188, 95th percentile 333  -> 336
#   bloom-560m: mean  20, 99th percentile  54  ->  64
# (measured on all splits, see outputs/tokenizer_comparison.md)
BASE_MODELS = {
    "distilgpt2": {
        "hf_name": "distilbert/distilgpt2",
        "attn": ["attn.c_attn"],
        "mlp": ["mlp.c_fc", "mlp.c_proj"],
        "max_len": 336,
    },
    "bloom-560m": {
        "hf_name": "bigscience/bloom-560m",
        "attn": ["self_attention.query_key_value"],
        "mlp": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
        "max_len": 64,
    },
}


@dataclass
class Config:
    # model
    base: str = "distilgpt2"      # key into BASE_MODELS
    targets: str = "attn"         # "attn" or "attn+mlp"
    # LoRA
    rank: int = 8
    alpha: float = 16.0           # the update is scaled by alpha / rank
    lora_dropout: float = 0.05
    # training
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-4
    warmup_frac: float = 0.05     # share of steps spent ramping the lr up from 0
    weight_decay: float = 0.0
    max_len: int | None = None    # prompt tokens (None = the base model's default);
                                  # longer headlines keep their beginning
    seed: int = 1337
    # quantization of the frozen base: None, "absmax4" or "nf4"
    quant: str | None = None
    quant_block: int = 64         # weights per scale in blockwise quantization
    # labels used for this run, filled in from the data at train time
    labels: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.max_len is None:
            self.max_len = BASE_MODELS[self.base]["max_len"]

    def target_modules(self) -> list[str]:
        spec = BASE_MODELS[self.base]
        return spec["attn"] + (spec["mlp"] if self.targets == "attn+mlp" else [])

    @property
    def hf_name(self) -> str:
        return BASE_MODELS[self.base]["hf_name"]

    def to_dict(self) -> dict:
        return asdict(self)
