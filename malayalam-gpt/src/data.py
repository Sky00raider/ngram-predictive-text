"""Turn the corpus into batches of (input, next-token target) tensors."""
from pathlib import Path

import torch

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"


def load_corpus(path: str | Path = CORPUS) -> str:
    return Path(path).read_text(encoding="utf-8")


def train_val_split(data: torch.Tensor, train_frac: float = 0.9):
    """First 90% of the token stream for training, last 10% for validation."""
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"):
    """Sample `batch_size` random windows of length `block_size`.

    x is a window, y is the same window shifted one step right, so at every
    position t the model sees x[:t+1] and must predict y[t] (the next token).
    """
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts])
    return x.to(device), y.to(device)
