"""Device detection, base-model presets and the run config.

One `Config` describes one fine-tuning run: which base model, how much of
LEDGAR to train on, and the Trainer settings. `src/train.py` builds it from
these defaults plus command-line overrides and saves it next to the run's
outputs, so every number in the README can be traced back to a config.
"""
from dataclasses import asdict, dataclass

import torch


def get_device() -> str:
    """Best available device: NVIDIA GPU, then Apple GPU, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Both models share GPT-2's tokenizer, so token lengths and max_len carry over.
#   distilgpt2:  6 layers, 82M parameters
#   gpt2:       12 layers, 124M parameters
# Learning rates are the usual ones for full fine-tuning of models this size;
# the smaller model tolerates a slightly larger step.
BASE_MODELS = {
    "distilgpt2": {"hf_name": "distilbert/distilgpt2", "lr": 1e-4},
    "gpt2": {"hf_name": "openai-community/gpt2", "lr": 5e-5},
}


@dataclass
class Config:
    # model
    base: str = "gpt2"              # key into BASE_MODELS
    # data
    # Training on all of it took 51 min for gpt2 on a free T4, so the default
    # is the whole deduplicated train split (59,080 clauses); a smaller number
    # draws a stratified subset that keeps the class imbalance.
    train_size: int = 60_000
    min_per_class: int = 10         # in a subset, every class keeps at least this many (or all it has)
    max_len: int = 384              # tokens per example: clause (cut at the end) + suffix + label;
                                    # 95% of clauses fit uncut (256 would fit 85%)
    eval_size: int = 2_000          # validation clauses scored each epoch to pick the best checkpoint
    # training (HF Trainer)
    epochs: int = 3
    batch_size: int = 16
    lr: float | None = None         # None = the base model's preset
    warmup_frac: float = 0.06       # share of steps spent ramping the lr up from 0
    weight_decay: float = 0.01
    seed: int = 1337

    def __post_init__(self):
        if self.lr is None:
            self.lr = BASE_MODELS[self.base]["lr"]

    @property
    def hf_name(self) -> str:
        return BASE_MODELS[self.base]["hf_name"]

    def run_name(self) -> str:
        return f"{self.base}_n{self.train_size // 1000}k"

    def to_dict(self) -> dict:
        return asdict(self)
