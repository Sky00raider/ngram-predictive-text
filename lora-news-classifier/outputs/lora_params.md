# LoRA trainable parameters

LoRA A/B only (the new label-token embedding rows add a few thousand more).

| base | total params | targets | layers wrapped | r=4 | r=8 | r=16 |
|---|---|---|---|---|---|---|
| distilgpt2 | 82M | attn | 6 | 73,728 (0.09%) | 147,456 (0.18%) | 294,912 (0.36%) |
| distilgpt2 | 82M | attn+mlp | 18 | 258,048 (0.32%) | 516,096 (0.63%) | 1,032,192 (1.26%) |
| bloom-560m | 559M | attn | 24 | 393,216 (0.07%) | 786,432 (0.14%) | 1,572,864 (0.28%) |
| bloom-560m | 559M | attn+mlp | 72 | 1,376,256 (0.25%) | 2,752,512 (0.49%) | 5,505,024 (0.98%) |
