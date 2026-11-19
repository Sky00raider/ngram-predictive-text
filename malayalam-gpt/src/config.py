"""Device detection and model-size presets.

The model size is picked to match the hardware, so one `python -m src.train`
gives a sensible ~1 hour run whether you have a GPU or only a CPU.
"""
from dataclasses import dataclass, asdict

import torch


def get_device() -> str:
    """Best available device: NVIDIA GPU, then Apple GPU, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # model shape
    block_size: int = 128   # context length (characters the model can look back over)
    n_embd: int = 128       # embedding size
    n_heads: int = 4        # attention heads per block (head_size = n_embd // n_heads)
    n_layers: int = 4       # transformer blocks
    dropout: float = 0.1
    # training
    batch_size: int = 32
    max_iters: int = 5000
    lr: float = 1e-3
    eval_interval: int = 250  # how often to measure train/val loss
    eval_iters: int = 50      # batches averaged per loss estimate
    seed: int = 1337

    def to_dict(self) -> dict:
        return asdict(self)


# One preset per device, each roughly an hour of training on that hardware.
PRESETS = {
    "cpu":  Config(),
    "mps":  Config(block_size=256, n_embd=256, n_heads=8, n_layers=6, batch_size=48, max_iters=6000, lr=6e-4),
    "cuda": Config(block_size=256, n_embd=384, n_heads=6, n_layers=6, batch_size=64, max_iters=8000, lr=3e-4, dropout=0.2),
}


def default_config(device: str | None = None) -> Config:
    return PRESETS[device or get_device()]
