"""LegalClauseTagger on a few hand-written clauses and real test clauses.

    python scripts/demo_tagger.py --model-dir models/gpt2_n60k --run gpt2_n60k

Real test clauses are picked from the run's saved test logits: the model's
most confident correct answer, two clauses it flags for review (one it got
wrong, one it got right), and its most confidently wrong answer. As a consistency check, the tagger's confidence on those clauses
(computed here, on CPU) must match the confidence from the logits saved on Colab.
Writes outputs/tagger_demo.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_labels, load_split  # noqa: E402
from src.metrics import softmax  # noqa: E402
from src.tagger import LegalClauseTagger  # noqa: E402

HANDWRITTEN = [
    "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, "
    "without regard to its conflict of laws principles.",
    "If any provision of this Agreement is held invalid or unenforceable, the remaining provisions shall "
    "continue in full force and effect.",
    "The Employee shall be entitled to twenty days of paid vacation per calendar year.",
    "The parties shall cooperate in good faith.",
]


def short(text: str, n: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/gpt2_n60k")
    p.add_argument("--run", default="gpt2_n60k")
    a = p.parse_args()

    tagger = LegalClauseTagger(a.model_dir, device="cpu")
    labels = load_labels()
    test = load_split("test")
    d = np.load(Path("outputs/runs") / a.run / "test_logits.npz")
    probs = softmax(d["logits"])
    pred, conf, target = probs.argmax(1), probs.max(1), d["target"]
    correct = pred == target

    picks = {
        "most confident, correct": int(np.argmax(np.where(correct, conf, -1))),
        "flagged for review, tag was wrong (caught)": int(np.argmin(np.abs(np.where(~correct, conf, 9) - 0.6))),
        "flagged for review, tag was right anyway": int(np.argmin(np.abs(np.where(correct, conf, 9) - 0.6))),
        "confidently wrong": int(np.argmax(np.where(~correct, conf, -1))),
    }
    lines = [f"# LegalClauseTagger demo (`{a.run}`, review threshold {tagger.threshold:.3f})\n",
             "## Hand-written clauses\n",
             "| clause | tag | confidence | review? | runner-up |", "|---|---|---|---|---|"]
    for clause, r in zip(HANDWRITTEN, tagger.classify_clauses(HANDWRITTEN)):
        lines.append(f"| {short(clause)} | {r['label']} | {r['confidence']:.3f} | "
                     f"{'needs review' if r['needs_review'] else 'no'} | {r['top3'][1][0]} ({r['top3'][1][1]:.3f}) |")

    lines += ["\n## Real test clauses\n",
              "| case | clause | true type | tag | confidence | review? |", "|---|---|---|---|---|---|"]
    max_diff = 0.0
    for case, i in picks.items():
        r = tagger.classify_clause(test[i]["text"])
        max_diff = max(max_diff, abs(r["confidence"] - float(conf[i])))
        lines.append(f"| {case} | {short(test[i]['text'])} | {labels[target[i]]} | {r['label']} | "
                     f"{r['confidence']:.3f} | {'needs review' if r['needs_review'] else 'no'} |")
    lines.append(f"\nConsistency check: tagger confidence on CPU vs saved Colab logits differ by at most "
                 f"{max_diff:.4f} (fp16 on the GPU vs fp32 here).")
    text = "\n".join(lines) + "\n"
    Path("outputs/tagger_demo.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
