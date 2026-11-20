import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import BenchConfig, model_size, val_windows  # noqa: E402
from src.p2 import GPT, Config, count_params, load_fp32_model  # noqa: E402


class TestP2Loading(unittest.TestCase):
    def test_checkpoint_loads_with_p2_parameter_count(self):
        model, cfg, info = load_fp32_model()
        self.assertEqual(count_params(model), 833_121)  # number in P2's README
        self.assertFalse(model.training)
        self.assertEqual(info["step"], 4750)

    def test_p2_alias_does_not_shadow_this_projects_src(self):
        import src
        import p2
        self.assertNotEqual(Path(src.__file__).parent, Path(p2.__file__).parent)


class TestBenchmark(unittest.TestCase):
    def test_val_windows_are_fixed_and_shifted_by_one(self):
        val = torch.arange(1000)
        bc = BenchConfig(val_batches=3, batch_size=4)
        a, b = list(val_windows(val, 16, bc)), list(val_windows(val, 16, bc))
        self.assertEqual(len(a), 3)
        for (x1, y1), (x2, _) in zip(a, b):
            self.assertTrue(torch.equal(x1, x2))
            self.assertTrue(torch.equal(x1[:, 1:], y1[:, :-1]))

    def test_model_size_counts_only_linear_layers_as_linear(self):
        model = GPT(Config(block_size=8, n_embd=16, n_heads=2, n_layers=1), vocab_size=10)
        s = model_size(model)
        self.assertLess(s["size_linear_bytes"], s["size_total_bytes"])
        linear_floats = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
                            for m in model.modules() if isinstance(m, nn.Linear))
        self.assertGreater(s["size_linear_bytes"], 4 * linear_floats)  # 4 bytes each + pickle overhead


if __name__ == "__main__":
    unittest.main()
