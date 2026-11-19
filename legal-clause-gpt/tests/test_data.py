import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import class_counts, dedupe, normalize, seen_mask, stratified_subset


def rows_for(counts: dict[str, int]) -> list[dict]:
    return [{"text": f"{label} clause number {i}", "label": label}
            for label, n in counts.items() for i in range(n)]


class TestDedupe(unittest.TestCase):
    def test_normalize_ignores_case_and_whitespace(self):
        self.assertEqual(normalize("  This  Agreement\nshall "), "this agreement shall")

    def test_drops_repeats_keeps_first_label(self):
        rows = [{"text": "Governed by NY law.", "label": "Governing Laws"},
                {"text": "governed  by ny law.", "label": "Applicable Laws"},
                {"text": "Notices in writing.", "label": "Notices"}]
        kept, report = dedupe(rows)
        self.assertEqual([r["label"] for r in kept], ["Governing Laws", "Notices"])
        self.assertEqual(report["repeats_dropped"], 1)
        self.assertEqual(report["texts_with_conflicting_labels"], 1)


class TestStratifiedSubset(unittest.TestCase):
    COUNTS = {"big": 900, "mid": 90, "rare": 10}

    def test_keeps_proportions_and_floor(self):
        rows = rows_for(self.COUNTS)
        sub = stratified_subset(rows, 100, min_per_class=5, seed=0)
        got = Counter(r["label"] for r in sub)
        self.assertEqual(got["big"], 90)
        self.assertEqual(got["mid"], 9)
        self.assertEqual(got["rare"], 5)      # 1 by proportion, raised to the floor

    def test_floor_never_exceeds_what_a_class_has(self):
        rows = rows_for({"big": 1000, "tiny": 3})
        got = Counter(r["label"] for r in stratified_subset(rows, 50, min_per_class=10, seed=0))
        self.assertEqual(got["tiny"], 3)

    def test_deterministic_and_no_repeats(self):
        rows = rows_for(self.COUNTS)
        a = stratified_subset(rows, 100, seed=1)
        b = stratified_subset(rows, 100, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(len({r["text"] for r in a}), len(a))

    def test_size_at_least_everything_returns_all(self):
        rows = rows_for(self.COUNTS)
        self.assertEqual(len(stratified_subset(rows, 10_000)), len(rows))


class TestHelpers(unittest.TestCase):
    def test_class_counts_in_label_order(self):
        rows = rows_for({"b": 2, "a": 1})
        self.assertEqual(class_counts(rows, ["a", "b", "c"]), [1, 2, 0])

    def test_seen_mask(self):
        train = [{"text": "Notices in writing.", "label": "Notices"}]
        rows = [{"text": "NOTICES in  writing.", "label": "Notices"},
                {"text": "Something else.", "label": "Notices"}]
        self.assertEqual(seen_mask(rows, train), [True, False])


if __name__ == "__main__":
    unittest.main()
