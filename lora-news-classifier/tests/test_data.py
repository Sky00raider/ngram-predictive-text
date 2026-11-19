import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data import ZWJ, label_names, load_split, normalize


class TestNormalize(unittest.TestCase):
    def test_legacy_chillu_becomes_atomic(self):
        # ല + virama + ZWJ  ->  ൽ (one code point)
        self.assertEqual(normalize("ജോലിയില്\u200d"), "ജോലിയിൽ")

    def test_no_zero_width_joiners_left(self):
        self.assertNotIn(ZWJ, normalize("കമ്ബോളങ്ങള്\u200d കരടിയുടെ"))

    def test_zero_width_space_dropped(self):
        self.assertEqual(normalize("സൂക്ഷിച്ചാൽ \u200bദു:ഖിക്കേണ്ട"), "സൂക്ഷിച്ചാൽ ദു:ഖിക്കേണ്ട")

    def test_digits_and_spaces(self):
        self.assertEqual(normalize("  ൨൦൧൮  ലോകകപ്പ് "), "2018 ലോകകപ്പ്")

    def test_keeps_latin_words(self):
        self.assertEqual(normalize("iPhone X വിപണിയിൽ"), "iPhone X വിപണിയിൽ")

    def test_idempotent(self):
        text = "കമ്ബോളങ്ങള്\u200d കരടിയുടെ പിടിയില്\u200d"
        self.assertEqual(normalize(normalize(text)), normalize(text))


class TestLoad(unittest.TestCase):
    def test_load_split_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.jsonl"
            path.write_text('{"text": "a  b", "label": "sports"}\n'
                            '{"text": "c", "label": "business"}\n', encoding="utf-8")
            rows = load_split("train", Path(d))
        self.assertEqual(rows, [("a b", "sports"), ("c", "business")])
        self.assertEqual(label_names(rows), ["business", "sports"])


class TestConfig(unittest.TestCase):
    def test_target_modules(self):
        self.assertEqual(Config(base="distilgpt2").target_modules(), ["attn.c_attn"])
        self.assertEqual(len(Config(base="bloom-560m", targets="attn+mlp").target_modules()), 3)

    def test_max_len_per_base(self):
        self.assertEqual(Config(base="distilgpt2").max_len, 336)
        self.assertEqual(Config(base="bloom-560m").max_len, 64)
        self.assertEqual(Config(base="distilgpt2", max_len=100).max_len, 100)


if __name__ == "__main__":
    unittest.main()
