import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bpe import CHUNK_RE, BPETokenizer
from src.tokenizer import CharTokenizer

# Repeated words so there are plenty of frequent pairs to merge.
SAMPLE = ("ക്ഷേത്രം തുറന്നു. ഇന്ന് 24 പേർക്ക് പ്രവേശനം!\n" * 20
          + "മലയാളം ഭാഷ പഠിക്കുന്നു, മലയാളം പുസ്തകം വായിക്കുന്നു.\n" * 20)


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = BPETokenizer.from_text(SAMPLE, vocab_size=120)

    def test_vocab_size(self):
        self.assertLessEqual(self.tok.vocab_size, 120)
        self.assertGreater(self.tok.vocab_size, len(self.tok.chars))

    def test_round_trip(self):
        text = "ക്ഷേത്രം തുറന്നു. ഇന്ന് 24 പേർക്ക്!\n"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_round_trip_whole_sample(self):
        self.assertEqual(self.tok.decode(self.tok.encode(SAMPLE)), SAMPLE)

    def test_chunks_cover_text(self):
        self.assertEqual("".join(CHUNK_RE.findall(SAMPLE)), SAMPLE)

    def test_save_load_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bpe.json"
            self.tok.save(path)
            loaded = BPETokenizer.load(path)
        self.assertEqual(loaded.chars, self.tok.chars)
        self.assertEqual(loaded.merges, self.tok.merges)
        self.assertEqual(loaded.encode(SAMPLE), self.tok.encode(SAMPLE))

    def test_merges_reduce_token_count(self):
        char_tok = CharTokenizer.from_text(SAMPLE)
        self.assertLess(len(self.tok.encode(SAMPLE)), len(char_tok.encode(SAMPLE)))

    def test_no_token_spans_two_words(self):
        # A token may start with a space (leading space of a word) but never
        # contain one anywhere else, so merges never cross a word boundary.
        for i in range(self.tok.vocab_size):
            self.assertNotIn(" ", self.tok.itos[i][1:], repr(self.tok.itos[i]))

    def test_unknown_char_raises(self):
        with self.assertRaises(ValueError):
            self.tok.encode("ക്ഷേത്രം €")


if __name__ == "__main__":
    unittest.main()
