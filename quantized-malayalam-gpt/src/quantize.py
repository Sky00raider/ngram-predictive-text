"""The three quantization methods from the course, applied to the P2 GPT.

    dynamic_int8(model)                 int8 weights, activations quantized on the fly per batch
    static_int8(model, calib_batches)   int8 weights and activations, ranges fixed by calibration
    qat_int8(model, train, ...)         fine-tune with fake quantization, then convert

Static PTQ and QAT quantize only the Linear layers. Each nn.Linear is wrapped as
QuantStub -> Linear -> DeQuantStub, so the tensor is int8 only inside the
Linear; everything between Linears (the attention matmuls q @ k.T and wei @ v,
the causal mask, softmax, LayerNorm, ReLU, residual adds, embeddings) runs in float.

Backend: this PyTorch build (2.15 CPU, Windows) ships only the oneDNN int8
engine; fbgemm and qnnpack are not compiled in. The CPU (Ryzen 7 7730U, Zen 3)
has AVX2 but no VNNI, so activations use reduce_range=True (7-bit range), as
fbgemm's own default does, to avoid overflow in the int8 multiply-accumulate.
"""
import copy

import torch
import torch.nn as nn
import torch.ao.quantization as tq

ENGINE = "onednn"

# static PTQ: histogram observer for activations, per-output-channel symmetric int8 weights
STATIC_QCONFIG = tq.QConfig(
    activation=tq.HistogramObserver.with_args(dtype=torch.quint8, reduce_range=True),
    weight=tq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
)

# QAT: the same int8 grids, simulated in float during training by FakeQuantize
QAT_QCONFIG = tq.QConfig(
    activation=tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8, reduce_range=True
    ),
    weight=tq.default_per_channel_weight_fake_quant,
)


def use_engine():
    torch.backends.quantized.engine = ENGINE


# ---------- dynamic ----------

def dynamic_int8(model: nn.Module) -> nn.Module:
    """Weights stored int8 ahead of time; each activation is scaled from its own min/max at run time."""
    use_engine()
    return tq.quantize_dynamic(copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8)


# ---------- the Linear-only wrapper used by static and QAT ----------

class QuantLinear(nn.Module):
    """float in -> quantize -> int8 Linear -> dequantize -> float out."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.quant = tq.QuantStub()
        self.linear = linear
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.linear(self.quant(x)))


def wrap_linears(model: nn.Module, qconfig: tq.QConfig) -> nn.Module:
    """Replace every nn.Linear with a QuantLinear carrying `qconfig`; nothing else gets a qconfig."""
    model.qconfig = None
    for parent in list(model.modules()):
        if isinstance(parent, QuantLinear):
            continue
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                wrapped = QuantLinear(child)
                wrapped.qconfig = qconfig
                setattr(parent, name, wrapped)
    return model


def activation_ranges(prepared: nn.Module) -> dict:
    """(min, max) each QuantStub's observer saw: the float range that will map onto 0..127."""
    out = {}
    for name, m in prepared.named_modules():
        if isinstance(m, QuantLinear):
            obs = m.quant.activation_post_process
            if hasattr(obs, "activation_post_process"):  # FakeQuantize wraps its observer
                obs = obs.activation_post_process
            out[name] = (obs.min_val.item(), obs.max_val.item())
    return out


# ---------- static PTQ ----------

@torch.no_grad()
def forward_from_embeddings(model, x: torch.Tensor):
    """Run the network starting from the embedding output (B, T, C) instead of token ids.

    Token ids can't be drawn from torch.randn, so the random calibration feeds
    Gaussian vectors where the token + position embeddings would be.
    """
    return model.lm_head(model.ln_f(model.blocks(x)))


@torch.no_grad()
def static_int8(model: nn.Module, calib_batches, from_embeddings: bool = False):
    """Post-training static quantization. Returns (int8 model, activation ranges seen in calibration).

    calib_batches: token-id tensors (B, T), or with from_embeddings=True float tensors (B, T, C).
    """
    use_engine()
    m = wrap_linears(copy.deepcopy(model).eval(), STATIC_QCONFIG)
    tq.prepare(m, inplace=True)  # inserts observers
    for x in calib_batches:      # calibration: only observe, no gradient, no weight change
        forward_from_embeddings(m, x) if from_embeddings else m(x)
    ranges = activation_ranges(m)
    tq.convert(m, inplace=True)  # observers -> scales/zero points; Linear -> quantized Linear
    return m, ranges


def real_calibration(train: torch.Tensor, block_size: int, n_batches: int = 16, batch_size: int = 32, seed: int = 1):
    """Random windows of real Malayalam from the training split (the val split stays untouched)."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        starts = torch.randint(len(train) - block_size, (batch_size,), generator=g)
        yield torch.stack([train[i : i + block_size] for i in starts])


def randn_calibration(block_size: int, n_embd: int, n_batches: int = 16, batch_size: int = 32, seed: int = 1):
    """The same number of batches, but standard-normal vectors in place of the embeddings."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        yield torch.randn(batch_size, block_size, n_embd, generator=g)


# ---------- QAT ----------

def prepare_qat_model(model: nn.Module) -> nn.Module:
    use_engine()
    m = wrap_linears(copy.deepcopy(model).train(), QAT_QCONFIG)
    return tq.prepare_qat(m, inplace=True)  # inserts FakeQuantize on weights and activations


def qat_finetune(prepared: nn.Module, train: torch.Tensor, block_size: int, steps: int = 300,
                 lr: float = 1e-4, batch_size: int = 32, freeze_observers_at: int = 200,
                 log_every: int = 50, seed: int = 2, log=print) -> list:
    """Short fine-tune with fake quantization in the forward pass.

    Rounding has zero gradient almost everywhere, so FakeQuantize uses the
    straight-through estimator: the backward pass treats it as identity, and the
    weights learn to sit well on the int8 grid. After `freeze_observers_at` steps
    the ranges stop moving so the last steps train against the final scales.
    Given a plain FP32 model it is an ordinary fine-tune (used as the QAT control).
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(prepared.parameters(), lr=lr)
    prepared.train()
    history = []
    for step in range(1, steps + 1):
        starts = torch.randint(len(train) - block_size, (batch_size,), generator=g)
        x = torch.stack([train[i : i + block_size] for i in starts])
        y = torch.stack([train[i + 1 : i + 1 + block_size] for i in starts])
        _, loss = prepared(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == freeze_observers_at:
            prepared.apply(tq.disable_observer)
        if step % log_every == 0 or step == 1:
            history.append((step, loss.item()))
            log(f"  qat step {step:4d}  train loss {loss.item():.4f}")
    return history


def convert_qat(prepared: nn.Module) -> nn.Module:
    use_engine()
    m = copy.deepcopy(prepared).eval()
    return tq.convert(m, inplace=True)
