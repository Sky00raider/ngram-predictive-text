import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import Config
from src.model import GPT, Head

CFG = Config(block_size=8, n_embd=16, n_heads=2, n_layers=2, dropout=0.0)


class TestModel(unittest.TestCase):
    def test_shapes_and_loss(self):
        model = GPT(CFG, vocab_size=10)
        x = torch.randint(10, (3, 8))
        logits, loss = model(x, x)
        self.assertEqual(logits.shape, (3, 8, 10))
        self.assertEqual(loss.shape, ())

    def test_causal_mask_hides_the_future(self):
        # changing a later token must not change the output at earlier positions
        torch.manual_seed(0)
        head = Head(CFG, head_size=4).eval()
        x = torch.randn(1, 8, 16)
        y = x.clone(); y[0, 5:] = torch.randn(3, 16)
        self.assertTrue(torch.allclose(head(x)[0, :5], head(y)[0, :5]))
        self.assertFalse(torch.allclose(head(x)[0, 5:], head(y)[0, 5:]))

    def test_generate_with_controls(self):
        model = GPT(CFG, vocab_size=10).eval()
        start = torch.zeros(1, 1, dtype=torch.long)
        for kw in [{}, {"temperature": 0.5}, {"top_k": 3}, {"top_p": 0.9}]:
            out = model.generate(start, max_new_tokens=12, **kw)  # longer than block_size
            self.assertEqual(out.shape, (1, 13))

    def test_top_k_1_is_greedy(self):
        torch.manual_seed(0)
        model = GPT(CFG, vocab_size=10).eval()
        start = torch.zeros(1, 1, dtype=torch.long)
        a = model.generate(start, 5, top_k=1)
        b = model.generate(start, 5, top_k=1)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
