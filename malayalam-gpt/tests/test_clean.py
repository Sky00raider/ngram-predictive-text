import unittest

from src.clean import clean_text

# A Malayalam sentence long enough to survive the 20-char minimum.
SENTENCE = "ഇത് ഒരു മലയാളം വാക്യം ആണ്"


class CleanTextTests(unittest.TestCase):
    def test_legacy_chillu_becomes_atomic(self):
        legacy = "ന\u0d4d\u200d"  # ന + virama + ZWJ
        out = clean_text(f"അവൻ {legacy} {SENTENCE}")
        self.assertIn("\u0d7b", out)
        self.assertNotIn("\u200d", out)
        self.assertNotIn("\u0d4d\u200d", out)

    def test_all_chillus(self):
        for base, atomic in [("ണ", "ൺ"), ("ന", "ൻ"), ("ര", "ർ"), ("ല", "ൽ"), ("ള", "ൾ"), ("ക", "ൿ")]:
            out = clean_text(f"{SENTENCE} {base}\u0d4d\u200d")
            self.assertIn(atomic, out)

    def test_zwnj_removed(self):
        out = clean_text(f"{SENTENCE} ക\u0d4d\u200cക")
        self.assertNotIn("\u200c", out)

    def test_malayalam_digits_to_ascii(self):
        out = clean_text(f"{SENTENCE} ൨൦൨൪")
        self.assertIn("2024", out)

    def test_english_heavy_line_dropped(self):
        english = "This is mostly an English sentence with ഒരു word"
        self.assertEqual(clean_text(english), "")

    def test_whitespace_collapsed(self):
        out = clean_text(f"  ഇത്   ഒരു \t മലയാളം   വാക്യം   ആണ്  \n\n\n")
        self.assertEqual(out, "ഇത് ഒരു മലയാളം വാക്യം ആണ്\n")

    def test_duplicates_and_short_lines_dropped(self):
        out = clean_text(f"{SENTENCE}\n{SENTENCE}\nചെറുത്\n")
        self.assertEqual(out, SENTENCE + "\n")

    def test_curly_quotes_and_dashes(self):
        out = clean_text(f"{SENTENCE} \u201cഅവൻ\u201d \u2013 ഇവൻ")
        self.assertIn('"അവൻ" - ഇവൻ', out)


if __name__ == "__main__":
    unittest.main()
