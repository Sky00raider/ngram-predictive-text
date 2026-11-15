import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ngram_lm import NgramLM, to_tokens  # noqa: E402
from ngram_lm.data import split_text  # noqa: E402
from ngram_lm.text import EOS  # noqa: E402

CORPUS = (
    "I drink coffee in the morning. I drink tea on the weekend. "
    "She reads a good book. She reads a film review. "
    "We walk on the park path. The morning train is quiet."
)


class TestTokens(unittest.TestCase):
    def test_curly_apostrophe(self):
        self.assertEqual(to_tokens("Don’t go"), ["don't", "go"])


class TestProbabilities(unittest.TestCase):
    def test_mle_counts(self):
        # with a tiny k the smoothed estimate is ~ the raw ratio: P(coffee | drink) = 1/2
        lm = NgramLM(2, k=1e-9).fit(CORPUS)
        self.assertAlmostEqual(lm.prob(("drink",), "coffee"), 0.5, places=6)

    def test_distributions_sum_to_one(self):
        for n in (1, 2, 3):
            for sm in ("addk", "witten-bell"):
                lm = NgramLM(n, k=0.1, smoothing=sm).fit(CORPUS)
                for ctx in [("i", "drink"), ("never", "seen"), ("<s>", "<s>")]:
                    ctx = ctx[len(ctx) - (n - 1) :] if n > 1 else ()
                    self.assertAlmostEqual(sum(lm.prob(ctx, w) for w in lm.vocab), 1.0, places=9)

    def test_unseen_ngram_has_nonzero_probability(self):
        for sm in ("addk", "witten-bell"):
            lm = NgramLM(2, smoothing=sm).fit(CORPUS)
            self.assertGreater(lm.prob(("coffee",), "train"), 0)

    def test_unknown_word_gets_mass(self):
        lm = NgramLM(2, min_count=2).fit(CORPUS)
        self.assertTrue(math.isfinite(lm.perplexity("zebras eat quokkas")))


class TestSuggest(unittest.TestCase):
    def test_suggestions(self):
        lm = NgramLM(2).fit(CORPUS)
        self.assertEqual(lm.suggest_next("i drink")[:2], ["coffee", "tea"])

    def test_never_suggests_end_of_sentence(self):
        lm = NgramLM(2).fit(CORPUS)
        self.assertNotIn(EOS, lm.suggest_next("good book", topk=10))

    def test_backoff_on_unseen_context(self):
        lm = NgramLM(3).fit(CORPUS)
        self.assertEqual(len(lm.suggest_next("xyzzy plugh", topk=3)), 3)

    def test_unigram_works(self):
        lm = NgramLM(1).fit(CORPUS)
        self.assertEqual(len(lm.suggest_next("anything", topk=2)), 2)


class TestComplete(unittest.TestCase):
    def test_partial_word_is_completed(self):
        lm = NgramLM(2).fit(CORPUS)
        self.assertEqual(lm.complete("i drink co")[0], "coffee")

    def test_trailing_space_suggests_next_word(self):
        lm = NgramLM(2).fit(CORPUS)
        self.assertEqual(lm.complete("i drink ")[:2], ["coffee", "tea"])

    def test_sentence_boundary_resets_context(self):
        lm = NgramLM(2).fit(CORPUS)
        self.assertEqual(lm.suggest_next("good book."), lm.suggest_next(""))


class TestGenerate(unittest.TestCase):
    def test_deterministic_per_seed(self):
        lm = NgramLM(2).fit(CORPUS)
        self.assertEqual(lm.generate(seed=3), lm.generate(seed=3))

    def test_respects_max_len(self):
        lm = NgramLM(2).fit(CORPUS)
        self.assertLessEqual(len(lm.generate(seed=1, max_len=4).split()), 4)


class TestPerplexity(unittest.TestCase):
    def test_in_sample_beats_out_of_sample(self):
        lm = NgramLM(2, k=0.1).fit(CORPUS)
        self.assertLess(lm.perplexity("I drink coffee in the morning"), lm.perplexity("train quiet the drink coffee"))

    def test_uniform_model_perplexity_is_vocab_size(self):
        # a unigram model with huge k is ~uniform over V, so perplexity ~ |V|
        lm = NgramLM(1, k=1e9).fit(CORPUS)
        self.assertAlmostEqual(lm.perplexity("coffee tea"), len(lm.vocab), delta=0.01 * len(lm.vocab))


class TestSplit(unittest.TestCase):
    def test_split_is_disjoint_and_reproducible(self):
        text = ". ".join(f"sentence number {i}" for i in range(100))
        a = split_text(text, seed=1)
        self.assertEqual(a, split_text(text, seed=1))
        self.assertEqual(sum(len(p.split(". ")) for p in a), 100)


if __name__ == "__main__":
    unittest.main()
