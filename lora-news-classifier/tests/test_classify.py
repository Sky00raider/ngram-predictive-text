import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from src.classify import HeadlineClassifier
from src.data import dedupe, unseen_mask
from src.lora import inject_lora

LABELS = ["business", "entertainment", "sports"]
HEADLINES = [
    "ഓഹരി വിപണി ഇന്ന് നേട്ടത്തിൽ",
    "സ്വർണവില വീണ്ടും കുറഞ്ഞു",
    "പുതിയ ചിത്രത്തിന്റെ ടീസർ പുറത്തിറങ്ങി",
    "നടന്റെ പിറന്നാൾ ആഘോഷം",
    "ലോകകപ്പ് ഫൈനൽ ഇന്ന്",
    "ഇന്ത്യക്ക് ആറ് വിക്കറ്റ് ജയം",
]
TARGETS = torch.tensor([0, 0, 1, 1, 2, 2])

_tokenizer = None


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    return _tokenizer


def tiny_classifier(max_len=64, lora=True):
    """Random 2-layer GPT-2 with distilgpt2's real vocabulary (so the real tokenizer fits)."""
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=128))
    if lora:
        inject_lora(model, ["attn.c_attn"], rank=4, alpha=8)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    return HeadlineClassifier(model, tokenizer(), LABELS, max_len).eval()


class TestEncode(unittest.TestCase):
    def test_last_position_is_end_of_prompt(self):
        clf = tiny_classifier()
        ids, mask, last = clf.encode(["ab", "a much longer headline here"])
        for i in range(2):
            self.assertEqual(mask[i, last[i]].item(), 1)
            self.assertEqual(ids[i, last[i]].item(), clf.suffix_ids[-1])
            # everything after `last` is padding
            self.assertEqual(mask[i, last[i] + 1:].sum().item(), 0)

    def test_truncation_keeps_start_and_suffix(self):
        clf = tiny_classifier(max_len=20)
        ids, mask, _ = clf.encode([HEADLINES[2] * 5])
        self.assertEqual(ids.shape[1], 20)
        full = tokenizer()(HEADLINES[2] * 5, add_special_tokens=False).input_ids
        n = 20 - len(clf.suffix_ids)
        self.assertEqual(ids[0, :n].tolist(), full[:n])
        self.assertEqual(ids[0, n:].tolist(), clf.suffix_ids)


class TestHead(unittest.TestCase):
    def test_label_ids_follow_vocab(self):
        clf = tiny_classifier()
        self.assertEqual(clf.label_ids, [50257, 50258, 50259])

    def test_rows_start_as_label_name_embedding(self):
        clf = tiny_classifier()
        emb = clf.model.get_input_embeddings().weight
        ids = tokenizer()(" sports", add_special_tokens=False).input_ids
        torch.testing.assert_close(clf.label_rows[2], emb[ids].mean(0))

    def test_logits_shape_and_label_part(self):
        clf = tiny_classifier()
        logits = clf(*clf.encode(HEADLINES))
        self.assertEqual(logits.shape, (6, 50257 + 3))
        # the old-vocab part is exactly the unmodified LM's next-token logits
        ids, mask, last = clf.encode(HEADLINES[:1])
        lm_logits = clf.model(input_ids=ids, attention_mask=mask).logits[0, last[0]]
        torch.testing.assert_close(logits[0, :50257], lm_logits)

    def test_padding_does_not_change_prediction(self):
        clf = tiny_classifier()
        alone = clf.label_logits([HEADLINES[0]])
        batched = clf.label_logits([HEADLINES[0], HEADLINES[2] * 3])
        torch.testing.assert_close(alone[0], batched[0], atol=1e-5, rtol=1e-5)

    def test_loss_is_cross_entropy_at_label_position(self):
        clf = tiny_classifier()
        logits = clf(*clf.encode(HEADLINES))
        expected = F.cross_entropy(logits, TARGETS + 50257)
        torch.testing.assert_close(clf.loss(HEADLINES, TARGETS), expected)


class TestTraining(unittest.TestCase):
    def test_only_lora_and_label_rows_get_gradients(self):
        clf = tiny_classifier()
        clf.train()
        clf.loss(HEADLINES, TARGETS).backward()
        with_grad = {n for n, p in clf.named_parameters() if p.grad is not None}
        self.assertIn("label_rows", with_grad)
        self.assertTrue(all(n == "label_rows" or n.endswith(("lora_A", "lora_B")) for n in with_grad))
        self.assertIsNone(clf.model.get_input_embeddings().weight.grad)

    def test_label_rows_alone_lower_the_loss(self):
        # With LoRA off only the 3 label rows train. The loss drops (the rows learn
        # to point at "a label comes next"), but a random model's hidden state at
        # the final ":" barely depends on the headline, so the rows alone can't
        # separate classes; making that state informative is LoRA's job.
        clf = tiny_classifier(lora=False)
        first, last = self._fit(clf)
        self.assertLess(last, first / 4)

    def test_lora_overfits_tiny_set(self):
        clf = tiny_classifier()
        first, last = self._fit(clf)
        self.assertLess(last, first / 4)
        clf.eval()
        self.assertEqual(clf.predict(HEADLINES), TARGETS.tolist())

    def _fit(self, clf, steps=60):
        opt = torch.optim.AdamW(clf.trainable_parameters(), lr=3e-2)
        clf.train()
        first = clf.loss(HEADLINES, TARGETS).item()
        for _ in range(steps):
            loss = clf.loss(HEADLINES, TARGETS)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return first, loss.item()


class TestDedupe(unittest.TestCase):
    def test_dedupe_and_conflicts(self):
        rows = [("a", "x"), ("b", "y"), ("a", "x"), ("c", "x"), ("c", "y")]
        self.assertEqual(dedupe(rows), [("a", "x"), ("b", "y")])

    def test_unseen_mask(self):
        self.assertEqual(unseen_mask([("a", "x"), ("z", "x")], [("a", "y")]), [False, True])


if __name__ == "__main__":
    unittest.main()
