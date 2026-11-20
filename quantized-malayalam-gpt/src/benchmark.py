"""One benchmark for every variant (FP32, dynamic, static x2, QAT): size, speed, val loss, sample.

Everything is fixed so the numbers are comparable across methods: the same
prompt, seed, token count, thread count and the same validation windows.
"""
import io
import platform
import statistics
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class BenchConfig:
    prompt: str = "കേരളത്തിൽ"   # "In Kerala", the prompt used throughout P2
    gen_tokens: int = 200        # new tokens per timed generation
    warmup_runs: int = 1
    speed_runs: int = 10
    num_threads: int = 1         # batch-1 generation on a tiny model: one thread gives the steadiest timings
    val_batches: int = 40        # 40 x 32 windows x 128 positions = 163,840 predicted tokens
    batch_size: int = 32
    temperature: float = 0.8
    top_k: int = 40
    seed: int = 0


# ---------- size ----------

# module types that hold a Linear layer's weights, before and after quantization
LINEAR_TYPES = [nn.Linear, torch.ao.nn.quantized.Linear]  # quantized.dynamic.Linear subclasses quantized.Linear


def _bytes(obj) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell()


def model_size(model: nn.Module) -> dict:
    """Bytes of the saved state dict: in total, and for the Linear layers alone.

    The rest (embeddings, LayerNorms, and each head's 128x128 causal-mask buffer)
    is left in float by every method, so the Linear-only number shows the compression.
    """
    sd = model.state_dict()
    linear_prefixes = tuple(
        name + "." for name, m in model.named_modules() if isinstance(m, tuple(LINEAR_TYPES))
    )
    linear_sd = {k: v for k, v in sd.items() if k.startswith(linear_prefixes)}
    return {"size_total_bytes": _bytes(sd), "size_linear_bytes": _bytes(linear_sd)}


# ---------- speed ----------

@torch.no_grad()
def tokens_per_sec(model, prompt_ids: torch.Tensor, bc: BenchConfig) -> dict:
    """Generate `gen_tokens` from the same prompt and seed; warm up, then average the timed runs."""
    def run():
        torch.manual_seed(bc.seed)
        start = time.perf_counter()
        model.generate(prompt_ids, bc.gen_tokens, temperature=bc.temperature, top_k=bc.top_k)
        return time.perf_counter() - start

    for _ in range(bc.warmup_runs):
        run()
    rates = [bc.gen_tokens / run() for _ in range(bc.speed_runs)]
    return {"tokens_per_sec": statistics.mean(rates), "tokens_per_sec_std": statistics.stdev(rates)}


@torch.no_grad()
def forward_tokens_per_sec(model, x: torch.Tensor, bc: BenchConfig) -> float:
    """Throughput of one full forward pass over a (batch, block) window: where int8 matmuls pay off.

    Batch-1 generation on a model this small is dominated by per-op overhead
    (16 heads, each a 128x32 matmul), so this second number isolates the arithmetic.
    """
    for _ in range(bc.warmup_runs):
        model(x)
    times = []
    for _ in range(bc.speed_runs):
        start = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - start)
    return x.numel() / statistics.mean(times)


# ---------- validation loss ----------

def val_windows(val: torch.Tensor, block_size: int, bc: BenchConfig):
    """Fixed (x, y) batches from the val split: the same windows for every method."""
    g = torch.Generator().manual_seed(bc.seed)
    starts = torch.randint(len(val) - block_size, (bc.val_batches, bc.batch_size), generator=g)
    for row in starts:
        x = torch.stack([val[i : i + block_size] for i in row])
        y = torch.stack([val[i + 1 : i + 1 + block_size] for i in row])
        yield x, y


@torch.no_grad()
def val_loss(model, val: torch.Tensor, block_size: int, bc: BenchConfig) -> float:
    losses = [model(x, y)[1].item() for x, y in val_windows(val, block_size, bc)]
    return statistics.mean(losses)


# ---------- sample ----------

@torch.no_grad()
def sample(model, tok, bc: BenchConfig) -> str:
    torch.manual_seed(bc.seed)
    idx = torch.tensor([tok.encode(bc.prompt)], dtype=torch.long)
    out = model.generate(idx, bc.gen_tokens, temperature=bc.temperature, top_k=bc.top_k)
    return tok.decode(out[0].tolist())


# ---------- all together ----------

def benchmark(name: str, model, tok, val: torch.Tensor, block_size: int, bc: BenchConfig = BenchConfig()) -> dict:
    torch.set_num_threads(bc.num_threads)
    model.eval()
    prompt_ids = torch.tensor([tok.encode(bc.prompt)], dtype=torch.long)
    result = {"method": name}
    result.update(model_size(model))
    result.update(tokens_per_sec(model, prompt_ids, bc))
    first_val_batch = next(val_windows(val, block_size, bc))[0]
    result["forward_tokens_per_sec"] = forward_tokens_per_sec(model, first_val_batch, bc)
    result["val_loss"] = val_loss(model, val, block_size, bc)
    result["sample"] = sample(model, tok, bc)
    result["bench"] = asdict(bc)
    result["env"] = {
        "torch": torch.__version__,
        "quant_engine": torch.backends.quantized.engine,
        "cpu": platform.processor(),
    }
    return result
