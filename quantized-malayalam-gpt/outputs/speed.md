# Speed, interleaved

7 rounds; each round times every variant once (rotated order): 3 generations of 200 tokens from `കേരളത്തിൽ`, and 3 forward passes over one 32x128 batch. 1 thread, engine onednn. Median over rounds, with min-max.

| method | generation tokens/sec | min-max | forward tokens/sec (batch) | min-max |
|---|---|---|---|---|
| fp32 | 148.9 | 118-151 | 26,285 | 18,472-26,960 |
| dynamic | 70.5 | 68-71 | 32,332 | 25,837-33,422 |
| static_real | 50.2 | 44-52 | 13,035 | 9,392-14,121 |
| static_randn | 48.9 | 45-51 | 13,830 | 13,401-14,888 |
| qat | 47.1 | 43-55 | 13,750 | 10,797-14,755 |
| fp32_finetuned | 151.4 | 113-161 | 26,739 | 21,889-28,707 |
