"""LegalClauseTagger: tag a contract clause and say how sure the tag is.

    from src.tagger import LegalClauseTagger

    tagger = LegalClauseTagger("path/to/models/gpt2_n60k")
    tagger.classify_clause("This Agreement shall be governed by the laws of the State of Delaware.")
    # {'label': 'Governing Laws', 'confidence': 0.998, 'needs_review': False,
    #  'top3': [('Governing Laws', 0.998), ('Applicable Laws', 0.001), ...]}

Confidence is the softmax over the 100 label-token scores at the end of the
prompt (src/recipe.py), i.e. the model's probability for the clause type it
picked. Below the review threshold the tag is still returned, but flagged
needs_review=True ("needs human review"). The threshold comes from the model
folder's tagger_config.json, where `python -m src.evaluate --model-dir ...`
stores the one chosen on validation for 95% accuracy; pass threshold= to override.

Command line:  python -m src.tagger --model-dir path/to/model "clause text" ["another clause" ...]
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import get_device
from src.recipe import ClauseEncoder, label_ids_of, label_logits, softmax

FALLBACK_THRESHOLD = 0.5


class LegalClauseTagger:
    def __init__(self, model_dir: str | Path, threshold: float | None = None, device: str | None = None):
        model_dir = Path(model_dir)
        self.labels: list[str] = json.loads((model_dir / "labels.json").read_text())
        conf = json.loads((model_dir / "tagger_config.json").read_text())
        self.threshold = threshold if threshold is not None else conf.get("threshold", FALLBACK_THRESHOLD)
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(self.device).eval()
        self.label_ids = label_ids_of(self.tokenizer, self.labels)
        self.encoder = ClauseEncoder(self.tokenizer, self.label_ids, conf["max_len"])

    @torch.no_grad()
    def classify_clauses(self, clauses: list[str], batch_size: int = 32) -> list[dict]:
        prompts, _ = self.encoder.prompts(clauses)
        probs = softmax(label_logits(self.model, prompts, self.label_ids, self.tokenizer.pad_token_id, batch_size))
        results = []
        for p in probs:
            top = p.argsort()[::-1][:3]
            confidence = float(p[top[0]])
            results.append({
                "label": self.labels[top[0]],
                "confidence": round(confidence, 4),
                "needs_review": confidence < self.threshold,
                "top3": [(self.labels[i], round(float(p[i]), 4)) for i in top],
            })
        return results

    def classify_clause(self, clause: str) -> dict:
        """Label, confidence, needs_review flag and the three most likely clause types."""
        return self.classify_clauses([clause])[0]


def main():
    p = argparse.ArgumentParser(description="Tag contract clauses with a LEDGAR clause type.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("clauses", nargs="+")
    a = p.parse_args()
    tagger = LegalClauseTagger(a.model_dir, a.threshold)
    print(f"review threshold {tagger.threshold:.3f}\n")
    for clause, r in zip(a.clauses, tagger.classify_clauses(a.clauses)):
        flag = "NEEDS HUMAN REVIEW" if r["needs_review"] else "ok"
        print(f"{clause[:90]}{'...' if len(clause) > 90 else ''}")
        print(f"  -> {r['label']}  (confidence {r['confidence']:.3f}, {flag})")
        print("     next: " + ", ".join(f"{name} {prob:.3f}" for name, prob in r["top3"][1:]) + "\n")


if __name__ == "__main__":
    main()
