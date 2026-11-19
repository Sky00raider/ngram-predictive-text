import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from transformers import BloomConfig, BloomForCausalLM, GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from src.lora import LoRALinear, count_parameters, inject_lora, lora_parameters


def tiny_gpt2():
    """A randomly initialised 2-layer GPT-2: same layer types as distilgpt2, no download."""
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=50, n_positions=16)).eval()


def tiny_bloom():
    torch.manual_seed(0)
    return BloomForCausalLM(BloomConfig(n_layer=2, hidden_size=32, n_head=2, vocab_size=50)).eval()


class TestLoRALinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4, 7, 12)

    def test_starts_as_identity_update(self):
        # B = 0, so the wrapped layer gives exactly the base layer's output.
        for base in (nn.Linear(12, 20), Conv1D(20, 12)):
            lora = LoRALinear(base, rank=4, alpha=8)
            self.assertTrue(torch.equal(lora(self.x), base(self.x)))
            self.assertTrue(torch.all(lora.lora_B == 0))
            self.assertFalse(torch.all(lora.lora_A == 0))

    def test_shapes_and_scaling(self):
        lora = LoRALinear(Conv1D(20, 12), rank=4, alpha=8)  # Conv1D(nf=out, nx=in)
        self.assertEqual((lora.in_features, lora.out_features), (12, 20))
        self.assertEqual(lora.lora_A.shape, (4, 12))
        self.assertEqual(lora.lora_B.shape, (20, 4))
        self.assertEqual(lora.scaling, 2.0)

    def test_forward_matches_merged_weight(self):
        # Adding delta_weight to W must give the same output as the LoRA path.
        base = nn.Linear(12, 20)
        lora = LoRALinear(base, rank=4, alpha=8)
        nn.init.normal_(lora.lora_B)
        merged = nn.Linear(12, 20)
        with torch.no_grad():
            merged.weight.copy_(base.weight + lora.delta_weight())
            merged.bias.copy_(base.bias)
        torch.testing.assert_close(lora(self.x), merged(self.x))

    def test_conv1d_matches_linear(self):
        # A Conv1D and a Linear holding the same (transposed) weight agree with LoRA on top.
        lin = nn.Linear(12, 20)
        conv = Conv1D(20, 12)
        with torch.no_grad():
            conv.weight.copy_(lin.weight.T)
            conv.bias.copy_(lin.bias)
        a, b = LoRALinear(lin, rank=4), LoRALinear(conv, rank=4)
        with torch.no_grad():
            b.lora_A.copy_(a.lora_A)
            nn.init.normal_(a.lora_B)
            b.lora_B.copy_(a.lora_B)
        torch.testing.assert_close(a(self.x), b(self.x))

    def test_only_lora_gets_gradients(self):
        base = nn.Linear(12, 20)
        lora = LoRALinear(base, rank=4)
        lora(self.x).sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNone(base.bias.grad)
        self.assertIsNotNone(lora.lora_B.grad)
        self.assertGreater(lora.lora_B.grad.abs().sum().item(), 0)
        # A's gradient is exactly zero on step one, because it is multiplied by B = 0.
        self.assertEqual(lora.lora_A.grad.abs().sum().item(), 0)


class TestInject(unittest.TestCase):
    def test_gpt2_attn_only(self):
        model = tiny_gpt2()
        ids = torch.randint(0, 50, (2, 10))
        before = model(ids).logits
        wrapped = inject_lora(model, ["attn.c_attn"], rank=4, alpha=8)
        self.assertEqual(wrapped, ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"])
        # still the same function right after injection
        torch.testing.assert_close(model(ids).logits, before)
        # trainable = 2 layers * r * (in + out) = 2 * 4 * (32 + 96)
        trainable, total = count_parameters(model)
        self.assertEqual(trainable, 2 * 4 * (32 + 96))
        self.assertEqual(set(n for n, p in model.named_parameters() if p.requires_grad),
                         set(lora_parameters(model)))

    def test_gpt2_attn_proj_not_matched(self):
        # "mlp.c_proj" must not also grab "attn.c_proj".
        model = tiny_gpt2()
        wrapped = inject_lora(model, ["mlp.c_fc", "mlp.c_proj"], rank=2, alpha=2)
        self.assertEqual(len(wrapped), 4)
        self.assertTrue(all(".mlp." in n for n in wrapped))

    def test_bloom_attn_and_mlp(self):
        model = tiny_bloom()
        targets = ["self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
        wrapped = inject_lora(model, targets, rank=4, alpha=8)
        self.assertEqual(len(wrapped), 6)
        trainable, _ = count_parameters(model)
        per_block = 4 * (32 + 96) + 4 * (32 + 128) + 4 * (128 + 32)
        self.assertEqual(trainable, 2 * per_block)

    def test_training_step_moves_only_lora(self):
        model = tiny_gpt2()
        inject_lora(model, ["attn.c_attn"], rank=4, alpha=8)
        frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        ids = torch.randint(0, 50, (2, 10))
        model.train()
        for _ in range(2):  # two steps: B moves on step 1, A on step 2
            loss = model(ids, labels=ids).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                self.assertTrue(torch.equal(p, frozen[n]), n)
        layer = model.transformer.h[0].attn.c_attn
        self.assertGreater(layer.lora_B.abs().sum().item(), 0)

    def test_no_match_raises(self):
        with self.assertRaises(ValueError):
            inject_lora(tiny_gpt2(), ["does.not.exist"], rank=4, alpha=8)


if __name__ == "__main__":
    unittest.main()
