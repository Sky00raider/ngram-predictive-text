# Sample quality (measured, no reading needed)

~3000 generated characters per row, prompts കേരളത്തിൽ, ഇന്ന്, സംസ്ഥാനത്ത്, പൊലീസ്, സിനിമ.
*Real words* = share of generated words that appear in the training text. *Script errors* = impossible sign sequences per 100 words.

| model | decoding | real words | script errors / 100 words |
|---|---|---|---|
| real validation text |  | 72% | 0.1 |
| char | temp 0.8, top-k 40 | 57% | 0.6 |
| char | temp 0.5 | 81% | 0.7 |
| char | temp 1.0 | 46% | 0.9 |
| char | temp 1.4 | 23% | 2.6 |
| bpe | temp 0.8, top-k 40 | 58% | 0.3 |
| bpe | temp 0.5 | 72% | 0.3 |
| bpe | temp 1.0 | 50% | 0.3 |
| bpe | temp 1.4 | 41% | 1.3 |
