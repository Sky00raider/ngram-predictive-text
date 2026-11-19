"""The recipe on a tiny random GPT-2 (real GPT-2 tokenizer), so it runs on CPU in seconds."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from src.recipe import (ClauseEncoder, LabelPositionCollator, add_label_tokens, label_ids_of,
                        label_logits, label_token, softmax)

LABELS = ["Governing Laws", "Indemnifications", "Notices", "Anti-Corruption Laws"]
CLAUSES = [
    "This Agreement shall be governed by the laws of the State of New York.",
    "The Company shall indemnify and hold harmless the Executive from any losses.",
    "All notices shall be in writing.",
    "Neither party shall pay any bribe to a government official.",
]
TARGETS = [0, 1, 2, 3]

def fresh():
    """A new tiny model + tokenizer with the label tokens added (add_tokens mutates the tokenizer)."""
    tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=128))
    original = model.get_input_embeddings().weight.detach().clone()
    label_ids = add_label_tokens(model, tok, LABELS)
    return model, tok, label_ids, original


class TestLabelTokens(unittest.TestCase):
    def test_token_names(self):
        self.assertEqual(label_token("Governing Laws"), "<governing_laws>")
        self.assertEqual(label_token("Anti-Corruption Laws"), "<anti_corruption_laws>")

    def test_tokens_added_and_embeddings_resized(self):
        model, tok, label_ids, original = fresh()
        V = original.shape[0]
        self.assertEqual(label_ids, list(range(V, V + len(LABELS))))
        self.assertEqual(model.get_input_embeddings().weight.shape[0], V + len(LABELS))
        self.assertEqual(label_ids_of(tok, LABELS), label_ids)
        for name, i in zip(LABELS, label_ids):   # each label is one unsplittable token
            self.assertEqual(tok(label_token(name), add_special_tokens=False).input_ids, [i])

    def test_output_layer_stays_tied(self):
        model, *_ = fresh()
        self.assertEqual(model.get_output_embeddings().weight.data_ptr(),
                         model.get_input_embeddings().weight.data_ptr())

    def test_rows_start_at_mean_name_embedding(self):
        model, tok, label_ids, original = fresh()
        name_ids = tok(" Governing Laws", add_special_tokens=False).input_ids
        torch.testing.assert_close(model.get_input_embeddings().weight[label_ids[0]],
                                   original[name_ids].mean(dim=0))
        torch.testing.assert_close(model.get_input_embeddings().weight[: original.shape[0]], original)

    def test_pad_token_fixed(self):
        model, tok, *_ = fresh()
        self.assertEqual(tok.pad_token_id, tok.eos_token_id)
        self.assertEqual(model.config.pad_token_id, tok.eos_token_id)


class TestEncodingAndLoss(unittest.TestCase):
    def setUp(self):
        self.model, self.tok, self.label_ids, _ = fresh()
        self.enc = ClauseEncoder(self.tok, self.label_ids, max_len=16)

    def test_suffix_length_matches_evaluate(self):
        from src.evaluate import SUFFIX_TOKENS
        self.assertEqual(len(self.enc.suffix_ids), SUFFIX_TOKENS)

    def test_truncation_keeps_suffix_and_label(self):
        long = " ".join(["word"] * 100)
        [ex] = self.enc.training_examples([long], [2])
        self.assertEqual(len(ex), 16)
        self.assertEqual(ex[-1], self.label_ids[2])
        self.assertEqual(ex[-1 - len(self.enc.suffix_ids):-1], self.enc.suffix_ids)
        prompts, n_tokens = self.enc.prompts([long])
        self.assertEqual(prompts[0], ex[:-1])
        self.assertEqual(n_tokens[0], 100)

    def test_collator_scores_only_the_label_position(self):
        examples = self.enc.training_examples(CLAUSES, TARGETS)
        batch = LabelPositionCollator(self.tok.pad_token_id)([{"input_ids": e} for e in examples])
        for i, e in enumerate(examples):
            n = len(e)
            scored = (batch["labels"][i] != -100).nonzero().flatten().tolist()
            self.assertEqual(scored, [n - 1])
            self.assertEqual(batch["labels"][i, n - 1].item(), self.label_ids[TARGETS[i]])
            self.assertEqual(batch["attention_mask"][i].sum().item(), n)

    def test_model_loss_is_cross_entropy_at_the_label_position(self):
        examples = self.enc.training_examples(CLAUSES, TARGETS)
        batch = LabelPositionCollator(self.tok.pad_token_id)([{"input_ids": e} for e in examples])
        out = self.model(**batch)
        last = batch["attention_mask"].sum(dim=1) - 1        # label position
        logits_before = out.logits[torch.arange(len(examples)), last - 1]   # prediction made at ":"
        targets = torch.tensor([self.label_ids[t] for t in TARGETS])
        torch.testing.assert_close(out.loss, F.cross_entropy(logits_before, targets))


class TestPrediction(unittest.TestCase):
    def test_label_logits_match_full_model_and_keep_order(self):
        model, tok, label_ids, _ = fresh()
        model.eval()
        enc = ClauseEncoder(tok, label_ids, max_len=64)
        prompts, _ = enc.prompts(CLAUSES)
        got = label_logits(model, prompts, label_ids, tok.pad_token_id, batch_size=3)
        self.assertEqual(got.shape, (len(CLAUSES), len(LABELS)))
        for i, p in enumerate(prompts):   # one at a time, no padding, full vocabulary
            with torch.no_grad():
                full = model(input_ids=torch.tensor([p])).logits[0, -1]
            np.testing.assert_allclose(got[i], full[label_ids].numpy(), rtol=1e-4, atol=1e-5)

    def test_softmax_rows_sum_to_one(self):
        p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
        np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0])
        self.assertAlmostEqual(p[1, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
