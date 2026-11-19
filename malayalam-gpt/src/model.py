"""A GPT written by hand: masked self-attention, transformer blocks, and sampling.

Shapes in comments: B = batch, T = time (tokens in the window), C = n_embd.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class Head(nn.Module):
    """One head of causal (masked) scaled dot-product self-attention."""

    def __init__(self, cfg: Config, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)    # what each token offers
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)  # what each token is looking for
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)  # what it passes on if attended to
        # lower-triangular mask: position t may only look at positions <= t
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)           # each (B, T, hs)
        # affinities between every pair of positions, scaled so softmax doesn't saturate
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5           # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # hide the future
        wei = F.softmax(wei, dim=-1)                                  # rows sum to 1
        wei = self.dropout(wei)
        return wei @ v                                                # weighted sum of values, (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated, then projected back to n_embd."""

    def __init__(self, cfg: Config):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_heads
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_heads)])
        self.proj = nn.Linear(cfg.n_heads * head_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: expand 4x, ReLU, project back. Attention mixes tokens; this 'thinks' per token."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block, pre-norm with residual connections."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # communicate between tokens
        x = x + self.ff(self.ln2(x))  # compute per token
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)          # what the token is
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)   # where it is
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)                     # scores for the next token

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)  # (B, T, C)
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)                                      # (B, T, vocab)
        if targets is None:
            return logits, None
        # next-token cross-entropy over every position in every window
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Autoregressive sampling: predict, sample one token, append, repeat."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])  # crop to the context window
            logits = logits[:, -1, :] / temperature          # last position only; <1 sharpens, >1 flattens
            if top_k is not None:
                # keep only the k highest-scoring tokens
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None:
                # nucleus: keep the smallest set of tokens whose probabilities sum to >= top_p
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                drop = cum - F.softmax(sorted_logits, dim=-1) >= top_p  # mass *before* this token already >= p
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
