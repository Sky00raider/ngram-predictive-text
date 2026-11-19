"""Load a trained model and generate Malayalam text.

    python -m src.generate --prompt "കേരളത്തിൽ"
    python -m src.generate --prompt "ഇന്ന്" --temperature 1.2 --top-p 0.9 --tokens 300
    python -m src.generate --run bpe --prompt "കേരളത്തിൽ"
    python -m src.generate --compare --prompt "കേരളത്തിൽ"   # writes outputs/<run>/decoding_comparison.md
"""
import argparse
from pathlib import Path

import torch

from .config import Config, get_device
from .model import GPT
from .tokenizer import CharTokenizer

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


class MiniGPT:
    """Load a trained GPT and generate text from a prompt: one object, one method."""

    def __init__(self, run_dir: str | Path = OUTPUTS / "char", device: str | None = None):
        run_dir = Path(run_dir)
        self.device = device or get_device()
        ckpt = torch.load(run_dir / "model.pt", map_location=self.device)
        if ckpt["tokenizer"] == "bpe":
            from .bpe import BPETokenizer
            self.tok = BPETokenizer.load(run_dir / "vocab.json")
        else:
            self.tok = CharTokenizer.load(run_dir / "vocab.json")
        self.model = GPT(Config(**ckpt["config"]), self.tok.vocab_size).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()  # turns dropout off

    def complete(self, prompt="\n", max_new_tokens=300, temperature=0.8, top_k=None, top_p=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        idx = torch.tensor([self.tok.encode(prompt)], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        return self.tok.decode(out[0].tolist())


# settings for the comparison file: the same prompt and seed, only the decoding changes
COMPARISON = [
    ("temperature 0.5", dict(temperature=0.5)),
    ("temperature 1.0", dict(temperature=1.0)),
    ("temperature 1.4", dict(temperature=1.4)),
    ("top-k 10 (temp 1.0)", dict(temperature=1.0, top_k=10)),
    ("top-p 0.9 (temp 1.0)", dict(temperature=1.0, top_p=0.9)),
]


def write_comparison(gpt: MiniGPT, prompt: str, tokens: int, path: Path, seed: int = 0):
    lines = [f"# Decoding comparison\n", f"Prompt: `{prompt}` | {tokens} new tokens | seed {seed} for every row\n"]
    for name, kw in COMPARISON:
        lines.append(f"## {name}\n\n{gpt.complete(prompt, tokens, seed=seed, **kw)}\n\n_English translation: TODO_\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="char", help="folder under outputs/ holding model.pt + vocab.json")
    p.add_argument("--prompt", default="കേരളത്തിൽ")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--compare", action="store_true", help="write the temperature/top-k/top-p comparison file")
    a = p.parse_args()

    gpt = MiniGPT(OUTPUTS / a.run)
    if a.compare:
        print("wrote", write_comparison(gpt, a.prompt, a.tokens, OUTPUTS / a.run / "decoding_comparison.md"))
    else:
        print(gpt.complete(a.prompt, a.tokens, a.temperature, a.top_k, a.top_p, a.seed))


if __name__ == "__main__":
    main()
