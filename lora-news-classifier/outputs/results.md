# Results

Test set: 630 headlines; "unseen" = the 524 test headlines that do not appear in train.

| run | base | targets | rank | quant | trainable params | adapter | val acc | test acc | test acc (unseen) | business (test) | entertainment (test) | sports (test) | s/step (CPU) | train time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bloom-560m_attn_r8 | bloom-560m | attn | 8 | - | 789,504 | 3,100 KB | 0.944 | **0.960** | 0.956 | 0.959 | 0.982 | 0.941 | 6.5 | 97 min |
| distilgpt2_attn_r8 | distilgpt2 | attn | 8 | - | 149,760 | 590 KB | 0.570 | **0.568** | 0.578 | 0.291 | 0.744 | 0.607 | 10.0 | 147 min |
