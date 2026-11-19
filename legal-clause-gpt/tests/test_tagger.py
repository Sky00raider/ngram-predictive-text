"""LegalClauseTagger on a tiny random model saved the same way src/train.py saves one."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from src.recipe import add_label_tokens
from src.tagger import LegalClauseTagger

LABELS = ["Governing Laws", "Notices", "Severability"]


class TestTagger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = Path(cls.dir.name)
        tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=32, n_positions=64))
        add_label_tokens(model, tok, LABELS)
        model.save_pretrained(d)
        tok.save_pretrained(d)
        (d / "labels.json").write_text(json.dumps(LABELS))
        (d / "tagger_config.json").write_text(json.dumps({"max_len": 64, "threshold": 0.4}))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_classify_clause(self):
        tagger = LegalClauseTagger(self.dir.name, device="cpu")
        r = tagger.classify_clause("All notices shall be in writing. " * 30)   # longer than max_len
        self.assertIn(r["label"], LABELS)
        self.assertTrue(0 < r["confidence"] <= 1)
        self.assertEqual(r["label"], r["top3"][0][0])
        self.assertEqual(r["needs_review"], r["confidence"] < 0.4)
        self.assertAlmostEqual(sum(p for _, p in r["top3"]), 1.0, places=3)   # only 3 labels here

    def test_threshold_override_and_batch(self):
        tagger = LegalClauseTagger(self.dir.name, threshold=1.0, device="cpu")
        rs = tagger.classify_clauses(["Short clause.", "Another, somewhat longer clause text."])
        self.assertEqual(len(rs), 2)
        self.assertTrue(all(r["needs_review"] for r in rs))


if __name__ == "__main__":
    unittest.main()
