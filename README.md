# Tinyverse projects

Six drills from Tinyverse, one folder each. Every project is self-contained: `cd` into it and follow its own README.

| # | Project | What it is |
|---|---|---|
| 1 | [predictive-text-engine](predictive-text-engine/) | Smoothed n-gram language model from scratch: next-word suggestions, autocomplete, generation, perplexity |
| 2 | [malayalam-gpt](malayalam-gpt/) | Character-level GPT written by hand in PyTorch and trained from scratch on Malayalam text |
| 3 | [lora-news-classifier](lora-news-classifier/) | LoRA written by hand, fine-tuning a frozen pretrained LM to classify Malayalam news headlines; distilgpt2 vs bloom-560m shows why the tokenizer matters |
| 4 | [legal-clause-gpt](legal-clause-gpt/) | GPT-2 fully fine-tuned to tag contract clauses with 100 LEDGAR types, with calibrated confidence and "needs human review" abstention |
| 5 | [quantized-malayalam-gpt](quantized-malayalam-gpt/) | Project 2's GPT quantized to int8 with dynamic, static (real vs random calibration) and QAT; size, CPU speed and val loss against FP32, with an FP32 fine-tune control for QAT |
| 6 | _coming soon_ | |
