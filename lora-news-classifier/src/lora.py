"""LoRA written by hand: a low-rank update on top of a frozen layer.

A pretrained layer computes  y = x W^T + b  with a big weight W (out x in).
Fine-tuning would change W directly. LoRA instead freezes W and learns a
small correction with the same shape, written as a product of two thin
matrices:

    y = x W^T + b  +  (alpha / r) * x A^T B^T
        \_ frozen _/    \___ trainable: A is (r x in), B is (out x r) ___/

With r = 8 and a 768 -> 2304 layer, W has 1.77M numbers but A and B together
only 8 * (768 + 2304) = 24.6k.

Initialisation (as in the LoRA paper):
  * A is random, B is all zeros, so B @ A = 0 and the wrapped model starts out
    computing exactly what the pretrained model computed.
  * The gradient for B is not zero (it depends on A x), so B moves on the
    first step, and after that A gets gradients too.
  * alpha / r keeps the size of the update roughly the same when you change r,
    so the learning rate doesn't need re-tuning for every rank.

The wrapped layer can be an `nn.Linear` or GPT-2's `Conv1D` (which stores its
weight transposed, as in x out). We never touch the base weight directly in
`forward`: we call the base layer and add the LoRA term, so both work.
"""
import math

import torch
from torch import nn


def layer_shape(layer: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) of an nn.Linear or a GPT-2 Conv1D."""
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    if type(layer).__name__ == "Conv1D":  # transformers.pytorch_utils.Conv1D
        n_in, n_out = layer.weight.shape  # stored (in, out): y = x @ W + b
        return n_in, n_out
    raise TypeError(f"LoRA does not know how to wrap {type(layer).__name__}")


class LoRALinear(nn.Module):
    """A frozen base layer plus a trainable low-rank update B @ A."""

    def __init__(self, base: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.in_features, self.out_features = layer_shape(base)
        self.rank = rank
        self.scaling = alpha / rank

        weight = next(base.parameters())  # match the base layer's device and dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=weight.device, dtype=weight.dtype))
        # Same random init that nn.Linear uses for its own weight.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen = self.base(x)
        # x A^T gives the r-dimensional bottleneck, then B^T maps it back up.
        update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return frozen + self.scaling * update

    def delta_weight(self) -> torch.Tensor:
        """The full (out x in) matrix this adapter adds to W: scaling * B @ A."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, scaling={self.scaling:g}"


def inject_lora(model: nn.Module, targets: list[str], rank: int, alpha: float, dropout: float = 0.0) -> list[str]:
    """Freeze the whole model, then swap every layer whose name ends with one
    of `targets` (e.g. "attn.c_attn") for a LoRALinear wrapping it.

    Returns the full names of the layers that were wrapped.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    # Collect first, replace after: changing modules while iterating is unsafe.
    chosen = [name for name, _ in model.named_modules()
              if any(name == t or name.endswith("." + t) for t in targets)]
    if not chosen:
        raise ValueError(f"No layers matched {targets}")

    for name in chosen:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), rank, alpha, dropout))
    return chosen


def lora_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Every LoRA A and B in the model, by name."""
    return {n: p for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
