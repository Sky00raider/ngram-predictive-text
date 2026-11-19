import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import get_batch, train_val_split
from src.tokenizer import CharTokenizer

SAMPLE = "ക്ഷേത്രം തുറന്നു. ഇന്ന് 24 പേർക്ക്!\n"


class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer.from_text(SAMPLE)

    def test_round_trip(self):
        self.assertEqual(self.tok.decode(self.tok.encode(SAMPLE)), SAMPLE)

    def test_one_token_per_code_point(self):
        # ക്ഷേത്രം is 3 visible units but 8 code points
        self.assertEqual(len(self.tok.encode("ക്ഷേത്രം")), 8)

    def test_unknown_char_raises(self):
        with self.assertRaises(ValueError):
            self.tok.encode("abc")

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            self.tok.save(Path(d) / "vocab.json")
            tok2 = CharTokenizer.load(Path(d) / "vocab.json")
        self.assertEqual(tok2.encode(SAMPLE), self.tok.encode(SAMPLE))


class TestBatching(unittest.TestCase):
    def test_targets_are_inputs_shifted_by_one(self):
        data = torch.arange(100)
        x, y = get_batch(data, block_size=8, batch_size=4)
        self.assertEqual(x.shape, (4, 8))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertTrue(torch.equal(y - x, torch.ones_like(x)))

    def test_split(self):
        train, val = train_val_split(torch.arange(100))
        self.assertEqual((len(train), len(val)), (90, 10))


if __name__ == "__main__":
    unittest.main()
