# Per-layer weight quantization error

mean |W - dequant(W_int8)| / mean |W| per Linear layer; key/query/value rows average the 4 heads. The last column is not a rounding error: it is how far QAT's fine-tune moved the float weights.

| layer | dynamic (per-tensor) | static (per-channel) | QAT (per-channel) | QAT drift from FP32 |
|---|---|---|---|---|
| `blocks.0.sa.heads.*.key` | 0.90% | 0.69% | 0.69% | 2.56% |
| `blocks.0.sa.heads.*.query` | 0.91% | 0.67% | 0.67% | 2.53% |
| `blocks.0.sa.heads.*.value` | 0.82% | 0.67% | 0.67% | 3.08% |
| `blocks.0.sa.proj` | 1.02% | 0.72% | 0.72% | 3.31% |
| `blocks.0.ff.net.0` | 1.07% | 0.68% | 0.68% | 2.61% |
| `blocks.0.ff.net.2` | 1.22% | 0.84% | 0.84% | 3.49% |
| `blocks.1.sa.heads.*.key` | 0.97% | 0.72% | 0.72% | 1.98% |
| `blocks.1.sa.heads.*.query` | 0.98% | 0.69% | 0.68% | 1.92% |
| `blocks.1.sa.heads.*.value` | 0.99% | 0.69% | 0.70% | 2.93% |
| `blocks.1.sa.proj` | 1.05% | 0.68% | 0.69% | 2.89% |
| `blocks.1.ff.net.0` | 0.97% | 0.70% | 0.69% | 2.13% |
| `blocks.1.ff.net.2` | 1.57% | 0.83% | 0.82% | 2.59% |
| `blocks.2.sa.heads.*.key` | 0.92% | 0.69% | 0.70% | 2.21% |
| `blocks.2.sa.heads.*.query` | 0.93% | 0.68% | 0.69% | 2.16% |
| `blocks.2.sa.heads.*.value` | 0.90% | 0.69% | 0.70% | 2.44% |
| `blocks.2.sa.proj` | 1.00% | 0.68% | 0.68% | 2.59% |
| `blocks.2.ff.net.0` | 1.09% | 0.70% | 0.70% | 1.98% |
| `blocks.2.ff.net.2` | 2.59% | 0.84% | 0.84% | 2.39% |
| `blocks.3.sa.heads.*.key` | 0.92% | 0.70% | 0.70% | 2.24% |
| `blocks.3.sa.heads.*.query` | 0.91% | 0.70% | 0.70% | 2.25% |
| `blocks.3.sa.heads.*.value` | 0.92% | 0.68% | 0.68% | 2.38% |
| `blocks.3.sa.proj` | 1.09% | 0.69% | 0.68% | 2.52% |
| `blocks.3.ff.net.0` | 1.09% | 0.71% | 0.71% | 1.87% |
| `blocks.3.ff.net.2` | 2.87% | 0.89% | 0.89% | 2.14% |
| `lm_head` | 0.98% | 0.65% | 0.64% | 2.24% |
| **mean over all 61 Linears** | **1.02%** | **0.70%** | **0.70%** | **2.42%** |
