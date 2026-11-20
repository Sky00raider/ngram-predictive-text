# LoRA from scratch: Malayalam news-headline classification

A LoRA adapter written by hand in PyTorch (no `peft`), used to fine-tune a frozen
pretrained language model to sort Malayalam news headlines into **business**,
**entertainment** or **sports**. It continues Project 2 ([malayalam-gpt](../malayalam-gpt/)):
there a Malayalam GPT was trained from scratch, here a pretrained model is adapted
by training **0.14% of its parameters**.

```python
from src.adapter import load_adapter

clf, cfg = load_adapter("outputs/runs/bloom-560m_attn_r8/adapter.pt", device="cpu")
headlines = ["ലോകകപ്പ് ഫൈനൽ ഇന്ന്", "ഓഹരി വിപണി ഇന്ന് നേട്ടത്തിൽ"]
[cfg.labels[i] for i in clf.predict(headlines)]   # ['sports', 'business']
```

**Headline result:** with the same recipe and the same LoRA settings, **bloom-560m reaches
96.0% test accuracy and distilgpt2 56.8%**. The difference is the tokenizer: distilgpt2 needs
188 tokens for an average headline (almost all of them byte fragments), bloom needs 20.

## What's in here

| Requirement | Where |
|---|---|
| Hand-written `LoRALinear`: frozen base, A random, B zeros, `alpha / r` scaling | `src/lora.py` (`LoRALinear`, `inject_lora`), tests in `tests/test_lora.py` |
| One label token per class, loss masked to the label position, argmax over label tokens | `src/classify.py` (`HeadlineClassifier`), training loop in `src/train.py` |
| Save adapter + new label-token rows together, reload onto a fresh base, verify | `src/adapter.py`, `scripts/verify_reload.py`, result in `outputs/reload_check.md` |
| distilgpt2 vs a multilingual base: tokens per headline and accuracy | `scripts/compare_tokenizers.py`, `outputs/tokenizer_comparison.md`, `outputs/results.md` |

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # (source .venv/bin/activate on Linux/macOS)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/download_data.py          # 0.6 MB of headlines into data/
python scripts/data_stats.py             # class balance, duplicates -> outputs/data_stats.md
python -m unittest discover -s tests     # 36 tests, tiny random models, ~5 s

python -m src.train --base bloom-560m    # one run (~1.6 h on an 8-core laptop CPU)
python -m src.train --base distilgpt2    # (~2.5 h)
python scripts/verify_reload.py outputs/runs/bloom-560m_attn_r8
python scripts/compare_tokenizers.py     # -> outputs/tokenizer_comparison.md
python scripts/results_table.py          # -> outputs/results.md

python scripts/run_queue.py stage5       # both runs + reload checks, unattended (overnight)
```

Everything here was trained on a laptop CPU (Ryzen 7 7730U, no GPU). The code also
runs on CUDA unchanged.

## How it works

### LoRA

A pretrained layer computes `y = x Wᵀ + b`. LoRA freezes `W` and learns a low-rank
correction of the same shape:

```
y = x Wᵀ + b  +  (alpha / r) · x Aᵀ Bᵀ        A: r × in (random),  B: out × r (zeros)
```

* **B starts at zero**, so right after injection the model computes exactly what the
  pretrained model did (tested). On the first step only B gets a gradient (A's is exactly
  zero, since it is multiplied by B = 0); after that both move.
* **`alpha / r`** keeps the size of the update roughly constant across ranks.
* `LoRALinear` calls the wrapped layer instead of reading its weight, so the same class wraps
  bloom's `nn.Linear` and GPT-2's `Conv1D` (which stores its weight transposed).
* `inject_lora` freezes the whole model and swaps each target layer, matched by name suffix
  (`attn.c_attn` for GPT-2, `self_attention.query_key_value` for BLOOM).

Trainable parameters, r = 8, attention only: distilgpt2 147k of 82M (0.18%), bloom-560m
786k of 559M (0.14%). Full table for other ranks: `outputs/lora_params.md`.

### The classification recipe

Each headline becomes the prompt `"<headline>\nCategory:"`. The model must produce one
of three **new tokens**, `<business>`, `<entertainment>`, `<sports>`, as the next token.

* **Label tokens.** The new ids are `V, V+1, V+2` (V = the old vocabulary size). In both
  models the output layer is tied to the input embeddings, so each label token's embedding
  row is also the vector the final hidden state is scored against. The rows start as the
  mean embedding of the label's English name and are trained along with LoRA.
  The three rows live in their own small tensor and their scores are appended to the
  model's logits. The maths is the same as resizing the embedding matrix, but the original
  250k × 1024 matrix stays out of the optimizer (AdamW would otherwise keep ~2 GB of state
  for it).
* **Loss masked to the label position.** Only the prediction made at the final `:` is
  scored: cross-entropy over the whole vocabulary plus the 3 label tokens.
* **Evaluation.** Argmax over the 3 label-token logits only, so the answer is always a class.

### Save and reload

`adapter.pt` holds only what was trained (every LoRA A and B, the 3 label rows) plus the
`Config` that says how to rebuild the frozen part. Loading builds a fresh pretrained base,
injects fresh LoRA layers (random A, zero B), creates fresh label rows, then copies the saved
tensors in, checking every name and shape.

## Dataset and cleaning

The iNLTK Malayalam headlines from IndicGLUE (`ai4bharat/indic_glue`, config `inltkh.ml`):
5,036 train / 630 validation / 630 test headlines, three classes, well balanced
(31% / 35% / 34% in train). Details: `outputs/data_stats.md`.

* **Cleaning** (`src/data.py`, adapted from Project 2): NFC, old-style chillu letters
  (consonant + virama + ZWJ, e.g. `ല്‍`) to atomic chillus (`ൽ`), drop ZWJ/ZWNJ/zero-width
  spaces, Malayalam digits to ASCII. **94% of headlines** were changed: they look identical
  on screen but tokenize differently.
* **Duplicates.** 462 train headlines are exact repeats, and **~17% of validation/test
  headlines also appear word-for-word in train**. Train is deduplicated (4,572 left;
  2 headlines with conflicting labels dropped). The official test split is kept, so numbers
  stay comparable with IndicGLUE, and accuracy is *also* reported on the 524 test headlines
  not seen in train ("unseen").

## Findings

All runs: LoRA r = 8, alpha = 16, attention projection only, AdamW lr 1e-3 with 5% warm-up
then linear decay, batch 16, 3 epochs (858 steps), seed 1337.

### 1. Results

| | bloom-560m | distilgpt2 |
|---|---|---|
| **test accuracy** | **0.960** | **0.568** |
| test accuracy, unseen headlines only | 0.956 | 0.578 |
| validation accuracy | 0.944 | 0.570 |
| business / entertainment / sports (test) | 0.959 / 0.982 / 0.941 | 0.291 / 0.744 / 0.607 |
| trainable parameters | 789,504 (0.14%) | 149,760 (0.18%) |
| adapter file | 3.1 MB | 0.6 MB |
| CPU time per step / whole run | 6.5 s / 97 min | 10.0 s / 147 min |

(Trainable counts include the 3 label rows: 3 × 1024 for bloom, 3 × 768 for distilgpt2.)

Validation accuracy across training (`outputs/runs/*/log.csv`):

| after epoch | bloom-560m | distilgpt2 |
|---|---|---|
| 0 (before training) | 0.357 | 0.356 |
| 1 | 0.900 | 0.367 |
| 2 | 0.943 | 0.543 |
| 3 | 0.944 | 0.570 |

Chance is about 0.33.

_Your analysis:_

### 2. Why the tokenizer matters for Malayalam

Tokens per headline, over all 6,296 cleaned headlines (`outputs/tokenizer_comparison.md`):

| tokenizer | vocab | mean tokens / headline | median | 95th pct | max | characters per token |
|---|---|---|---|---|---|---|
| distilgpt2 | 50,257 | **188.4** | 174 | 333 | 962 | 0.37 |
| bloom-560m | 250,680 | **20.1** | 19 | 34 | 98 | 3.44 |

The same headline, split by each (`·` = a token that is only part of one character's
UTF-8 bytes and cannot be shown on its own):

**പുതിയ ഫീച്ചറുമായി ഫേസ്ബുക്ക്; അഡ്മിനുകൾക്ക് പോസ്റ്റുകൾ ഇനി എളുപ്പത്തിൽ തയ്യാറാക്കാം** (83 characters, "Facebook with a new feature; admins can now prepare posts easily")

- distilgpt2, 233 tokens: `·|·|·|·|·|·|·|·|·|·|·|·|·|·|·| |·|·|·|·|…` (every token is a byte fragment; the only readable ones are spaces and `;`)
- bloom-560m, 20 tokens: `പുതിയ| ഫ|ീച്ച|റ|ുമായി| ഫേസ്ബ|ുക്ക്|;| അഡ|്മ|ിന|ുകൾക്ക്| പോസ്റ്റ|ുകൾ| ഇനി| എളുപ്പ|ത്തിൽ| തയ്യാറ|ാക്ക|ാം`

GPT-2's byte-level BPE was trained on English web text and has almost no merges for
Malayalam, so each 3-byte Malayalam letter becomes 2–3 tokens. bloom was pretrained on
Malayalam among other Indic languages, and its vocabulary holds real word pieces
(`ഫേസ്ബ|ുക്ക്` "Facebook", `പോസ്റ്റ|ുകൾ` "posts").

Practical consequences seen here:

* **Accuracy.** Same recipe, same LoRA, 96.0% vs 56.8%.
* **Compute.** distilgpt2 is 7× smaller but each step took 1.5× longer, because every batch
  is ~9× more tokens and attention cost grows with sequence length.
* **Context.** To leave 95% of headlines uncut distilgpt2 needs `max_len` 336 (bloom: 64).
  GPT-2's 1,024-token window holds only ~340 Malayalam characters.

_Your analysis:_

### 3. Save → reload onto a fresh base

`outputs/reload_check.md`:

| run | at the end of training | fresh base + reloaded adapter | fresh base, untrained adapter (control) | max logit difference |
|---|---|---|---|---|
| bloom-560m_attn_r8 | 0.9603 | 0.9603 | 0.3905 | 0.0 |
| distilgpt2_attn_r8 | 0.5683 | 0.5683 | 0.3794 | 0.0 |

Every one of the 630 predictions is identical after reloading, and the logits match exactly.
The control shows the check can fail: without the adapter the same base is at chance.

_Your analysis:_

## Limitations / next steps

* One seed per configuration, so differences under about a point are within noise.
* bloom saw far more data in pretraining than distilgpt2 (and is 7× larger), so the comparison
  is "tokenizer + pretraining language coverage", not the tokenizer alone.
* Not done (optional extras from the brief): a rank 4/8/16 × attention vs attention + MLP
  ablation, QLoRA (quantizing the base to 4-bit before vs after training) and NF4.
  The ablation needs no new code (`--rank`, `--targets attn+mlp`; `scripts/run_queue.py` can
  queue the grid, about one night on CPU). QLoRA would need a quantized frozen layer under
  `LoRALinear`; `Config.quant` is a placeholder for it.

## Layout

```
src/        config.py (presets), data.py (cleaning, dedup), lora.py (LoRALinear),
            classify.py (label-token recipe), train.py, adapter.py (save / load)
scripts/    download_data.py, data_stats.py, count_params.py, compare_tokenizers.py,
            verify_reload.py, results_table.py, run_queue.py
outputs/    data_stats.md, lora_params.md, tokenizer_comparison.md, results.md,
            reload_check.md, runs/<run>/ (config, log, metrics, adapter.pt)
tests/      unit tests (tiny random models, no download except the GPT-2 tokenizer)
```

Data: IndicGLUE iNLTK headlines (AI4Bharat), downloaded, not committed.
Base models: [distilgpt2](https://huggingface.co/distilbert/distilgpt2) and
[bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m), downloaded on first use.
