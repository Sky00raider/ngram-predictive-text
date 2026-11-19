import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.metrics import (at_threshold, confused_pairs, coverage_curve, fit_temperature, macro_f1,
                         nll, per_class, reliability, softmax, spearman, threshold_for_accuracy)


class TestAccuracyAndF1(unittest.TestCase):
    def test_per_class_and_macro_f1(self):
        target = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 1, 1, 1, 2, 0])
        s = per_class(pred, target, 3)
        np.testing.assert_allclose(s["recall"], [0.5, 1.0, 0.5])
        np.testing.assert_allclose(s["precision"], [0.5, 2 / 3, 1.0])
        f1 = [0.5, 0.8, 2 / 3]
        np.testing.assert_allclose(s["f1"], f1)
        self.assertAlmostEqual(macro_f1(pred, target, 3), np.mean(f1))

    def test_macro_f1_ignores_classes_absent_everywhere(self):
        target = np.array([0, 1])
        self.assertAlmostEqual(macro_f1(target.copy(), target, 50), 1.0)

    def test_confused_pairs_counts_both_directions(self):
        target = np.array([0, 0, 0, 1, 2, 2])
        pred = np.array([1, 1, 0, 0, 0, 2])
        pairs = confused_pairs(pred, target)
        self.assertEqual(pairs[0], (0, 1, 2, 1))   # 0->1 twice, 1->0 once
        self.assertEqual(pairs[1], (0, 2, 0, 1))


class TestAbstention(unittest.TestCase):
    conf = np.array([0.99, 0.9, 0.8, 0.6, 0.4])
    correct = np.array([1, 1, 0, 1, 0], dtype=bool)

    def test_coverage_curve(self):
        cov, acc, thr = coverage_curve(self.conf, self.correct)
        np.testing.assert_allclose(cov, [0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_allclose(acc, [1, 1, 2 / 3, 0.75, 0.6])
        np.testing.assert_allclose(thr, self.conf)

    def test_threshold_maximises_coverage_at_target(self):
        self.assertEqual(threshold_for_accuracy(self.conf, self.correct, 0.75), 0.6)
        self.assertEqual(threshold_for_accuracy(self.conf, self.correct, 1.0), 0.9)
        r = at_threshold(self.conf, self.correct, 0.6)
        self.assertEqual((r["coverage"], r["accuracy"], r["abstained"]), (0.8, 0.75, 1))

    def test_unreachable_target_answers_nothing(self):
        t = threshold_for_accuracy(np.array([0.9, 0.5]), np.array([False, True]), 0.99)
        self.assertEqual(at_threshold(np.array([0.9, 0.5]), np.array([False, True]), t)["coverage"], 0.0)


class TestCalibration(unittest.TestCase):
    def test_perfectly_calibrated_bins_have_zero_ece(self):
        conf = np.array([0.95] * 20 + [0.55] * 20)
        correct = np.array([True] * 19 + [False] + [True] * 11 + [False] * 9)
        r = reliability(conf, correct)
        self.assertAlmostEqual(r["ece"], 0.0)
        self.assertEqual(r["count"][9], 20)
        self.assertEqual(r["count"][5], 20)

    def test_overconfidence_shows_up(self):
        r = reliability(np.full(10, 0.9), np.array([True] * 5 + [False] * 5))
        self.assertAlmostEqual(r["ece"], 0.4)

    def test_temperature_recovers_overconfidence(self):
        rng = np.random.default_rng(0)
        true_logits = rng.normal(size=(4000, 5)) * 2
        target = np.array([rng.choice(5, p=p) for p in softmax(true_logits)])
        sharp = true_logits * 3                       # same argmax, 3x too confident
        t = fit_temperature(sharp, target)
        self.assertAlmostEqual(t, 3.0, delta=0.4)
        self.assertLess(nll(sharp, target, t), nll(sharp, target, 1.0))


class TestSpearman(unittest.TestCase):
    def test_monotone_and_ties(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 1000]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(spearman([1, 1, 2], [1, 1, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
