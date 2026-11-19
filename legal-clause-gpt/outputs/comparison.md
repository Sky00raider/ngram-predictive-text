# distilgpt2 vs gpt2

| | `gpt2_n60k` | `distilgpt2_n60k` |
|---|---|---|
| parameters | 124M | 82M |
| training time (T4) | 50.7 min | 33.5 min |
| test accuracy | 87.3% | 85.5% |
| test macro-F1 | 80.6% | 77.3% |
| test accuracy, unseen clauses | 87.0% | 85.3% |
| coverage at 95% accuracy | 79.8% | 74.2% |
| coverage at 98% accuracy | 59.4% | 50.0% |
| ECE (raw softmax) | 4.7% | 4.3% |
| ECE after temperature scaling | 1.2% (T=1.38) | 1.3% (T=1.23) |
| rank corr. train frequency vs per-class accuracy | +0.48 | +0.48 |
