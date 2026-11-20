# Calibration: real text vs torch.randn

(min, max) recorded by each Linear's input observer during static calibration. Only head 0 is shown; key/query/value share one input, so they see the same range. Width ratio = randn range / real range: below 1 means randn calibration would clip real activations.

| layer | input comes from | real | randn | width ratio |
|---|---|---|---|---|
| `blocks.0.sa.heads.0.query` | LayerNorm ln1 | [-4.26, 4.74] | [-5.04, 5.29] | 1.15 |
| `blocks.0.sa.proj` | concatenated attention heads | [-3.49, 3.49] | [-3.86, 3.81] | 1.10 |
| `blocks.0.ff.net.0` | LayerNorm ln2 | [-3.82, 4.12] | [-3.99, 4.82] | 1.11 |
| `blocks.0.ff.net.2` | ReLU | [0.00, 5.07] | [0.00, 4.26] | 0.84 |
| `blocks.1.sa.heads.0.query` | LayerNorm ln1 | [-5.05, 5.06] | [-5.39, 5.54] | 1.08 |
| `blocks.1.sa.proj` | concatenated attention heads | [-5.26, 4.58] | [-4.39, 4.81] | 0.93 |
| `blocks.1.ff.net.0` | LayerNorm ln2 | [-5.86, 6.21] | [-5.68, 6.15] | 0.98 |
| `blocks.1.ff.net.2` | ReLU | [0.00, 9.12] | [0.00, 7.71] | 0.85 |
| `blocks.2.sa.heads.0.query` | LayerNorm ln1 | [-4.70, 5.04] | [-5.10, 5.28] | 1.07 |
| `blocks.2.sa.proj` | concatenated attention heads | [-5.37, 4.60] | [-4.34, 4.01] | 0.84 |
| `blocks.2.ff.net.0` | LayerNorm ln2 | [-7.36, 6.18] | [-6.41, 6.21] | 0.93 |
| `blocks.2.ff.net.2` | ReLU | [0.00, 10.46] | [0.00, 8.11] | 0.78 |
| `blocks.3.sa.heads.0.query` | LayerNorm ln1 | [-5.04, 4.70] | [-5.20, 5.13] | 1.06 |
| `blocks.3.sa.proj` | concatenated attention heads | [-5.50, 5.24] | [-3.90, 4.08] | 0.74 |
| `blocks.3.ff.net.0` | LayerNorm ln2 | [-7.89, 6.58] | [-6.99, 6.94] | 0.96 |
| `blocks.3.ff.net.2` | ReLU | [0.00, 11.76] | [0.00, 10.70] | 0.91 |
| `lm_head` | final LayerNorm ln_f | [-8.67, 6.01] | [-8.24, 6.64] | 1.01 |
