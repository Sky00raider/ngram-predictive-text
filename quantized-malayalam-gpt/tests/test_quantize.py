import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import quantize as Q  # noqa: E402
from src.p2 import GPT, Config  # noqa: E402


def tiny_gpt():
    torch.manual_seed(0)
    return GPT(Config(block_size=16, n_embd=32, n_heads=2, n_layers=2, dropout=0.0), vocab_size=20).eval()


def tokens(batch=4, t=16):
    return torch.randint(0, 20, (batch, t), generator=torch.Generator().manual_seed(1))


class TestWrapLinears(unittest.TestCase):
    def test_every_linear_is_wrapped_and_nothing_else(self):
        m = Q.wrap_linears(tiny_gpt(), Q.STATIC_QCONFIG)
        n_linear = 2 * (3 * 2 + 1 + 2) + 1  # per block: 2 heads x (k, q, v) + proj + 2 ff; plus lm_head
        wrapped = [x for x in m.modules() if isinstance(x, Q.QuantLinear)]
        self.assertEqual(len(wrapped), n_linear)
        bare = [x for x in m.modules() if isinstance(x, nn.Linear) and not any(x is w.linear for w in wrapped)]
        self.assertEqual(bare, [])
        # only the wrappers carry a qconfig, so LayerNorm / Embedding stay float
        self.assertIsNone(getattr(m.blocks[0].ln1, "qconfig", None))
        self.assertIsNone(getattr(m.token_embedding, "qconfig", None))

    def test_wrapping_alone_does_not_change_outputs(self):
        m = tiny_gpt()
        x = tokens()
        before = m(x)[0]
        after = Q.wrap_linears(m, Q.STATIC_QCONFIG)(x)[0]  # stubs are identity until convert
        self.assertTrue(torch.allclose(before, after))


class TestMethods(unittest.TestCase):
    def assertClose(self, q_model, fp32, tol):
        x = tokens()
        ref, out = fp32(x)[0], q_model(x)[0]
        rel = ((out - ref).abs().mean() / ref.abs().mean()).item()
        self.assertLess(rel, tol)

    def test_dynamic_is_close_to_fp32(self):
        fp32 = tiny_gpt()
        self.assertClose(Q.dynamic_int8(fp32), fp32, 0.05)

    def test_static_is_close_to_fp32_and_attention_stays_float(self):
        fp32 = tiny_gpt()
        q, ranges = Q.static_int8(fp32, [tokens(8) for _ in range(4)])
        self.assertClose(q, fp32, 0.1)
        head = q.blocks[0].sa.heads[0]
        self.assertIsInstance(head.key.linear, torch.ao.nn.quantized.Linear)
        self.assertEqual(head.tril.dtype, torch.float32)
        self.assertTrue(all(lo < hi for lo, hi in ranges.values()))

    def test_randn_calibration_runs_from_embeddings(self):
        fp32 = tiny_gpt()
        q, _ = Q.static_int8(fp32, Q.randn_calibration(16, 32, n_batches=2, batch_size=4), from_embeddings=True)
        self.assertClose(q, fp32, 0.15)

    def test_qat_trains_and_converts(self):
        fp32 = tiny_gpt()
        prepared = Q.prepare_qat_model(fp32)
        data = torch.randint(0, 20, (500,))
        hist = Q.qat_finetune(prepared, data, 16, steps=4, batch_size=4, freeze_observers_at=2, log=lambda s: None)
        self.assertEqual([s for s, _ in hist], [1])
        q = Q.convert_qat(prepared)
        self.assertIsInstance(q.lm_head.linear, torch.ao.nn.quantized.Linear)
        self.assertEqual(q(tokens())[0].shape, (4, 16, 20))


if __name__ == "__main__":
    unittest.main()
