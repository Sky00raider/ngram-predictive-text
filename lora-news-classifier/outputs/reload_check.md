## reload_check_distilgpt2

adapter file: 590 KB (base model distilbert/distilgpt2 not included) | 630 test headlines

| | test accuracy |
|---|---|
| at the end of training | 0.4444 |
| fresh base + reloaded adapter | 0.4444 |
| fresh base, untrained adapter (control) | 0.3794 |

max |logit difference| after reload: 0.00e+00 | identical predictions: True | **MATCH**

## bloom-560m_attn_r8

adapter file: 3,100 KB (base model bigscience/bloom-560m not included) | 630 test headlines

| | test accuracy |
|---|---|
| at the end of training | 0.9603 |
| fresh base + reloaded adapter | 0.9603 |
| fresh base, untrained adapter (control) | 0.3905 |

max |logit difference| after reload: 0.00e+00 | identical predictions: True | **MATCH**

## distilgpt2_attn_r8

adapter file: 590 KB (base model distilbert/distilgpt2 not included) | 630 test headlines

| | test accuracy |
|---|---|
| at the end of training | 0.5683 |
| fresh base + reloaded adapter | 0.5683 |
| fresh base, untrained adapter (control) | 0.3794 |

max |logit difference| after reload: 0.00e+00 | identical predictions: True | **MATCH**

