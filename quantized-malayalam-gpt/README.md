# Quantized Malayalam GPT: dynamic, static and QAT int8 on a GPT built from scratch

The Malayalam character-level GPT from Project 2 ([malayalam-gpt](../malayalam-gpt/)) quantized to int8
with the three methods from the course (dynamic quantization, static post-training quantization and
quantization-aware training) and compared against the FP32 original on CPU. Across the projects:
P2 **builds** a model, P3 **adapts** one, and this project **compresses** one.

```python
from src.p2 import load_fp32_model
from src import quantize as Q

fp32, cfg, _ = load_fp32_model()      # P2's trained checkpoint, loaded read-only
int8 = Q.dynamic_int8(fp32)           # or Q.static_int8(fp32, calib_batches), or the QAT path
```

**Headline result:** dynamic int8 shrinks the Linear weights **3.7x (3.23 MB -> 0.88 MB)** and the
whole file **2.1x (4.42 MB -> 2.07 MB)**, and val loss rises by **0.0012** (1.5087 -> 1.5099).
Every int8 variant runs slower than FP32 when generating one token at a time. The model is so small
that the time goes on per-operation overhead, and int8 adds more operations, not fewer.

## What's in here

| Requirement | Where |
|---|---|
| Load the trained P2 checkpoint and model code | `src/p2.py`: imports P2's `src/` package read-only under the alias `p2` |
| FP32 baseline: size, tokens/sec, val loss on the P2 val split, sample | `src/benchmark.py`, `scripts/evaluate.py --method fp32` |
| Dynamic int8 on `nn.Linear` | `dynamic_int8` in `src/quantize.py` |
| Static PTQ on Linear layers only, attention matmuls and softmax in float | `QuantLinear`, `wrap_linears`, `static_int8` in `src/quantize.py` |
| Calibrate on real Malayalam vs `torch.randn`, compare | `real_calibration`, `randn_calibration`; `outputs/calibration_ranges.md` (Finding 4) |
| QAT: start from FP32, short fake-quant fine-tune, convert | `prepare_qat_model`, `qat_finetune`, `convert_qat` in `src/quantize.py` (Finding 5) |
| One results table | `scripts/results_table.py` -> `outputs/results.md` (below) |
| Extra: per-layer quantization error | `scripts/layer_error.py` -> `outputs/layer_error.md` (Finding 6) |
| Extra: same prompt and seed, samples from every variant | `outputs/samples_side_by_side.md` (Finding 7) |

## Run it

Commands run from `quantized-malayalam-gpt/`. This project reads `../malayalam-gpt/outputs/char/`
(checkpoint and vocab, committed) and `../malayalam-gpt/data/corpus.txt` (not committed; create it
with P2's `scripts/download_data.py` and `scripts/prepare_data.py`). It never writes to P2.

```bash
python -m venv .venv && .venv/Scripts/activate      # (source .venv/bin/activate on Linux/macOS)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/evaluate.py --method fp32             # each writes outputs/results/<method>.json, seconds each
python scripts/evaluate.py --method dynamic
python scripts/evaluate.py --method static_real
python scripts/evaluate.py --method static_randn
python scripts/evaluate.py --method qat              # 300-step fake-quant fine-tune, ~4 min on 8 threads
python scripts/evaluate.py --method fp32_finetuned   # control for QAT, ~2 min
#   add --reuse to qat / fp32_finetuned to load the committed outputs/models/*.pt instead of training
python scripts/speed.py                              # all six variants interleaved, ~15 min -> outputs/speed.md
python scripts/layer_error.py                        # -> outputs/layer_error.md
python scripts/results_table.py                      # -> outputs/results.md, samples_side_by_side.md, calibration_ranges.md
python -m unittest discover -s tests                 # 10 tests, ~4 s
```

## Setup: backend, settings and what is measured

* **CPU only.** PyTorch's int8 kernels run on CPU only. Machine: AMD Ryzen 7 7730U (Zen 3, 8 cores, AVX2, no VNNI), torch 2.15.0+cpu on Windows.
* **Backend: oneDNN, not fbgemm.** On an x86 CPU the course's choice is fbgemm (qnnpack is for ARM). This Windows build of torch compiles in only one int8 engine: `torch.backends.quantized.supported_engines == ['onednn']`, and setting `fbgemm`, `qnnpack` or `x86` raises "not supported". oneDNN warns that without VNNI its default `reduce_range=False` can overflow the int8 multiply-accumulate, so activations use `reduce_range=True` (a 7-bit range, 0..127), as fbgemm's own default does. Weights are per-output-channel symmetric int8 for static and QAT. `quantize_dynamic` uses its default per-tensor weights.
* **Deprecation.** Every `torch.ao.quantization` call prints a warning that eager-mode quantization is moving to `torchao`. The APIs still work in 2.15, and this project uses them as the course does.
* **Model.** P2's char model: 4 blocks, 4 heads, `n_embd` 128, context 128, vocab 97, 833,121 parameters, checkpoint from step 4750. It has 61 Linear layers: per block, 4 heads x (key, query, value), the attention `proj` and two feed-forward layers, plus `lm_head`.
* **Size** is the saved `state_dict`. "Linear weights" counts only the Linear layers. The rest (embeddings, LayerNorms, and each head's 128x128 causal-mask buffer, 16 x 64 KB = 1 MB of the 4.4 MB) stays float under every method, which is why the whole file shrinks less than the Linear layers do.
* **Val loss**: 40 fixed batches of 32 x 128 windows from P2's val split (last 10% of the corpus), **the same windows for every row**, so differences between rows are paired, not sampling noise. On these windows the FP32 checkpoint scores 1.5087. P2 logged 1.4976, its best estimate over random batches.
* **Speed**: 1 thread, batch 1, 200 new tokens from `കേരളത്തിൽ` with seed 0, temperature 0.8, top-k 40. "Forward" is one 32 x 128 batch, to show raw matmul throughput. Run separately, the same FP32 model measured anywhere from 105 to 163 tokens/sec, because the laptop's clock drifts with temperature and power state. So `scripts/speed.py` loads all six variants in one process, times each once per round in a rotated order for 7 rounds, and reports the median.

## Results

| method | size (MB) | Linear weights (MB) | generation tok/s | forward tok/s | val loss | Δ val loss |
|---|---|---|---|---|---|---|
| FP32 baseline (P2 checkpoint) | 4.42 | 3.23 | 149 | 26,285 | 1.5087 | |
| Dynamic int8 | **2.07** | **0.88** | 71 | **32,332** | 1.5099 | +0.0012 |
| Static int8, calibrated on real text | 2.23 | 0.98 | 50 | 13,035 | 1.5119 | +0.0032 |
| Static int8, calibrated on `torch.randn` | 2.23 | 0.98 | 49 | 13,830 | 1.5112 | +0.0024 |
| QAT int8 (300-step fake-quant fine-tune) | 2.23 | 0.98 | 47 | 13,750 | 1.4827 | -0.0260 (**+0.0047 vs control**) |
| Control: FP32, same 300-step fine-tune | 4.42 | 3.23 | 151 | 26,739 | **1.4780** | -0.0307 |

Samples (seed 0; first ~70 characters, full text in [`outputs/samples_side_by_side.md`](outputs/samples_side_by_side.md)):

| method | sample |
|---|---|
| FP32 | കേരളത്തിൽനിന്നും സംഭവം കഴിഞ്ഞ ആരോപണം യുവാഗ്രഹിക്കുന്നതും ഒന്നും സുമങ്ക… |
| Dynamic | കേരളത്തിൽനിന്നും സംഭവം കാട്ട് ആരോപണം യുവാഗ്രഹിക്കുന്നത്. ഒന്നാം സുമങ്ക… |
| Static, real | കേരളത്തിൽനിന്നും സംഭവം കഴിഞ്ഞ ആരോപണം യുവാഗ്രഹിക്കുന്നതും ഒന്നും സുമങ്ക… |
| Static, randn | കേരളത്തിൽനിന്നും സംഭവം കാട്ട് ആരോപണം യുവാഗ്രഹിക്കുന്നതു തിരുവനന്തപുരം … |
| QAT | കേരളത്തിൽനിന്നും സംഭവം കുടുംബശ്രീയ ഏജൻ. ബഗാനിലെ പത്ര പല തിന്റെ വിജയങ്ങ… |
| Control | കേരളത്തിൽനിന്നും സംഭവം കുടുംബശ്രീയ ഏജൻ. ബഗാനിലെ പത്ര പല തിന്റെ വിജയങ്ങ… |

## Why attention stays in float

Static PTQ and QAT quantize **only the Linear layers**. Each `nn.Linear` is wrapped as
`QuantStub -> Linear -> DeQuantStub` (`QuantLinear` in `src/quantize.py`), so a tensor is int8 only
inside a Linear and is float again before anything else touches it. That leaves in float:

* **`q @ k.T` and `wei @ v`.** These multiply two *activations*, not an activation by a stored
  weight. Both operands change with every input, so neither has a range that can be calibrated once,
  and eager-mode PyTorch has no quantized module for a plain `@` anyway: it would need
  `FloatFunctional` wrappers and a requantize after every product.
* **The causal mask and softmax.** The mask writes `-inf` above the diagonal, and int8 has no
  infinity. Softmax is an exponential: scores a few units apart become probabilities orders of
  magnitude apart, and a 256-level (here 128-level) uniform grid can't keep both the near-zero
  weights and the large ones. Small changes in the scores shift which characters the model
  attends to, so errors here spread through every later layer.
* **LayerNorm, residual adds, embeddings.** LayerNorm divides by a per-token standard deviation, and
  the residual stream adds up every block's output, so its range grows with depth. Both are cheap,
  so quantizing them would save almost no time and add rounding error.

The Linear layers hold almost all the weights (802,017 of 833,121 parameters, 3.21 of 3.33 MB) and almost all
the multiply-adds, so quantizing only them gets nearly all of the size reduction while leaving
the fragile parts exact. Dynamic quantization ends up with the same split, because
`quantize_dynamic` only replaces `nn.Linear`.

> **My analysis:** _…_

## Findings

### 1. Size: 3.7x on the Linear layers, 2.0-2.1x on the file

An int8 weight takes 1 byte instead of 4, so the ideal is 4x. Dynamic reaches 3.69x on the Linear
layers. The rest is the bias (kept float) and packing overhead. Static and QAT reach 3.28x, because
per-channel quantization stores a scale and a zero point for every output channel, and 61 small
layers make that metadata noticeable. On the whole file the 1 MB of causal-mask buffers and the
float embeddings and LayerNorms don't shrink, which caps it at about 2x.

> **My analysis:** _…_

### 2. Accuracy: post-training int8 costs at most 0.003 val loss

Dynamic +0.0012, static +0.0024 to +0.0032. On a loss of 1.51, that is at most 0.2%. The int8
samples match FP32 for the first 24 characters (71 for static real), then diverge, as expected
once a slightly different probability picks a different character. They stay just as plausible.

> **My analysis:** _…_

### 3. Speed: int8 is slower for generation and faster only for dynamic on batches

| | generation, batch 1 | forward, 32 x 128 batch |
|---|---|---|
| dynamic vs FP32 | 0.47x | **1.23x** |
| static / QAT vs FP32 | 0.32-0.34x | 0.50-0.53x |

Generating one token runs 61 Linear layers on a (1, ≤128, 128) input, and each head's Linear is only
128 x 32. At that size a call costs more in dispatch, memory layout and launch overhead than in
arithmetic, so int8's cheaper multiply-adds don't help. The overhead is also higher:

* **Dynamic** measures the activation's min/max, quantizes, runs the int8 matmul and dequantizes,
  at every Linear call. With a full batch there is enough arithmetic to pay for that: +23%.
* **Static and QAT** add a separate `Quantize` and `DeQuantize` module around each of the 61 Linears.
  Key, query and value each quantize the same LayerNorm output on their own, so every block quantizes
  one tensor 12 times. Nothing is fused, and oneDNN on a CPU without VNNI has no fast int8 dot product,
  so even the batched forward runs at half FP32's speed.

Fixes this project does not apply: fuse each head's k/q/v into one Linear (one quantize instead of
three), quantize the residual stream once per block instead of around every Linear, and use a
VNNI-capable CPU or the fbgemm engine.

> **My analysis:** _…_

### 4. Calibration: `torch.randn` was almost as good as real text, because of LayerNorm

Val loss: real-text calibration 1.5119, `torch.randn` 1.5112 (`randn` was even 0.0007 better, which is
within what different clipping points can do). Token ids can't be drawn from a Gaussian, so the randn
calibration feeds `torch.randn(32, 128, 128)` where the token + position embeddings would be
(`forward_from_embeddings`). The recorded input ranges ([`outputs/calibration_ranges.md`](outputs/calibration_ranges.md)) show why it works:

| Linear input comes from | layers | randn range / real range |
|---|---|---|
| a LayerNorm (`ln1` -> k/q/v, `ln2` -> `ff.net.0`, `ln_f` -> `lm_head`) | 53 of 61 | 0.93-1.15 |
| the concatenated attention heads (`sa.proj`) | 4 | 1.10, 0.93, 0.84, **0.74** |
| the ReLU (`ff.net.2`) | 4 | 0.84, 0.85, 0.78, 0.91 |

53 of the 61 Linears read a LayerNorm output. LayerNorm rescales every token to mean 0 and variance 1
before applying its learned scale and shift, so whatever comes in (real text or noise), the range
coming out is set mostly by the learned scale and shift. Only the layers fed by attention or the ReLU
see something input-dependent, and there randn's ranges are up to 26% narrower in the deeper
blocks, so real activations get clipped. That was too little to show in the loss here. A model without
pre-norm, or one with real outlier features, would make random calibration much riskier.

> **My analysis:** _…_

### 5. QAT: better than FP32, but only because of the extra training

QAT's val loss (1.4827) is **below** the FP32 baseline (1.5087). Read alone, that looks like
quantization helping. The control row trains the FP32 model with the identical 300 steps (same seed,
batches and AdamW lr 1e-4) without fake quantization and reaches 1.4780. P2 trained at a constant
lr of 1e-3 with no decay, so 300 steps at a 10x lower lr anneals the checkpoint. Against its fair
reference, **QAT costs +0.0047**, more than static PTQ costs (+0.0024 to +0.0032).

QAT exists to recover accuracy that PTQ loses. Here PTQ loses almost nothing, so there is nothing to
recover. In 300 steps the weights don't settle onto the int8 grid either (Finding 6: QAT's rounding
error is the same 0.70% as static's). Fake quantization adds noise to the forward pass, and the
fine-tune has to work through it.

> **My analysis:** _…_

### 6. Extra: per-layer weight error

`mean |W - dequant(W_int8)| / mean |W|` for every Linear ([`outputs/layer_error.md`](outputs/layer_error.md),
k/q/v averaged over the 4 heads):

| | mean over 61 Linears | worst layer |
|---|---|---|
| dynamic (per-tensor) | 1.02% | `blocks.3.ff.net.2`: 2.87% |
| static (per-channel) | 0.70% | `blocks.3.ff.net.2`: 0.89% |
| QAT (per-channel, vs its own fine-tuned weights) | 0.70% | `blocks.3.ff.net.2`: 0.89% |
| QAT fine-tune drift from FP32 (not rounding) | 2.42% | |

* **Per-channel beats per-tensor.** One scale for a whole matrix is set by its largest weight, so the
  other rows get a coarse grid. The second feed-forward layers in blocks 2 and 3 (`ff.net.2`, 512 -> 128) have
  the widest weight spread and suffer most per-tensor (2.6-2.9%). Per-channel stays at 0.64-0.89% everywhere.
* **Weight error doesn't predict loss.** Dynamic has the *largest* weight error but the *smallest*
  loss increase, because its activations are quantized with their exact min/max for every input.
  Static's activations use one fixed range from calibration. On this model activation error matters
  more than weight error.
* **The fine-tune moved weights 3.5x more than rounding does** (2.42% vs 0.70%), which matches
  Finding 5: QAT's val loss is mostly the effect of training, not of quantization.

> **My analysis:** _…_

### 7. Extra: samples side by side

Same prompt, seed 0, temperature 0.8, top-k 40, 200 new characters for all variants, in
[`outputs/samples_side_by_side.md`](outputs/samples_side_by_side.md). Because the seed is shared, each
variant copies FP32 character for character until its quantized probabilities first send the
sampler to a different character (after 24 characters for most, 71 for static real). QAT and its
FP32 control agree with each other for the first 100 characters, since both went through the same
300 fine-tuning steps on the same batches; what separates them is fake quantization during training and int8 rounding at inference.

> **My analysis:** _…_

## Limitations / next steps

* **One small model, one CPU.** At 833k parameters and 128-wide layers, per-op overhead dominates and int8
  can't show a speed-up. A model with 1024-wide layers and a longer context would put these methods
  in a regime where the arithmetic matters.
* **oneDNN without VNNI, no fbgemm.** The numbers reflect this build. An older torch wheel with fbgemm,
  or a CPU with VNNI, would likely change the speed column.
* **QAT is short.** 300 steps, about 4 minutes. A longer run might recover the +0.0047, but on a model where
  PTQ loses 0.2% there is little point.
* **No fusion.** Each head's k/q/v are separate Linears, quantized separately. Fusing them, or quantizing
  once per block, would cut most of the static-mode overhead.
* **Next**: quantize the BPE model too (`../malayalam-gpt/outputs/bpe`), try int8 for the embeddings, and
  try torchao's pt2e flow, which replaces eager-mode quantization.

## Layout

```
src/        p2.py (loads P2 read-only), benchmark.py (size, speed, val loss, sample), quantize.py (dynamic, static, QAT)
scripts/    evaluate.py (build + benchmark one method), speed.py (interleaved timing), layer_error.py, results_table.py
tests/      10 unit tests (P2 loading, benchmark windows, wrapping, each method on a tiny GPT)
outputs/    results.md, speed.md, layer_error.md, calibration_ranges.md, samples_side_by_side.md
  results/  one JSON per method (all metrics, sample, activation ranges, QAT loss history)
  models/   qat_prepared.pt (QAT fine-tune before convert), fp32_finetuned.pt (control), with their logs
```
