"""The classification recipe: GPT-2 answers with one label token.

Each clause becomes the sequence

    <clause tokens> "\nClause type:" <label token>

and the model is trained, as an ordinary causal language model, to produce
the label token after the prompt.

  * Label tokens.  One new vocabulary entry per class (<governing_laws>,
    <indemnifications>, ...), 100 in all, added to the tokenizer as special
    tokens so they are never split. The model's embedding matrix is resized
    to match (`resize_token_embeddings`); GPT-2 ties its output layer to the
    input embeddings, so each new row is both how the token is read and the
    vector the final hidden state is scored against.
    Each new row starts as the mean embedding of the label's English name
    (" Governing Laws" -> its GPT-2 tokens), not noise: the pretrained model
    already knows what those words mean, which helps the rare classes that
    get few gradient updates.

  * Pad token.  GPT-2 has none, so the end-of-text token doubles as padding.
    Padding is on the right, masked out of attention, and never a target.

  * Loss masked to the label position.  `labels` is -100 (ignored) everywhere
    except the final position, which holds the label token. The model shifts
    labels by one internally, so the single prediction that is scored is the
    one made at the last prompt token (":"). Cross-entropy is over the full
    vocabulary plus the 100 label tokens, exactly as in language modelling.

  * Evaluation.  We run the prompt (without the label), take the hidden state
    at its last token, and score it against the 100 label rows only. Argmax
    gives the class; softmax over those 100 scores gives the confidence.
    The model can therefore only ever answer with a valid clause type.

Clauses longer than max_len keep their beginning; the suffix and label are
always kept, so truncation can never cut off the thing being predicted.
"""
import re
from dataclasses import dataclass

import numpy as np
import torch

PROMPT_SUFFIX = "\nClause type:"


def label_token(name: str) -> str:
    """'Governing Laws' -> '<governing_laws>'."""
    return "<" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ">"


def add_label_tokens(model, tokenizer, labels: list[str]) -> list[int]:
    """Pad-token fix, one new token per label, resized embeddings. Returns the label token ids."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Name-based starting rows, computed from the original embeddings.
    embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        init = torch.stack([
            embeddings[tokenizer(" " + name, add_special_tokens=False).input_ids].mean(dim=0)
            for name in labels
        ])

    tokens = [label_token(name) for name in labels]
    added = tokenizer.add_tokens(tokens, special_tokens=True)
    if added != len(labels):
        raise ValueError(f"expected {len(labels)} new label tokens, tokenizer added {added}")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    label_ids = tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        model.get_input_embeddings().weight[label_ids] = init.to(embeddings.dtype)
    return label_ids


def label_ids_of(tokenizer, labels: list[str]) -> list[int]:
    """Label token ids of a tokenizer that already has them (e.g. a saved model)."""
    return tokenizer.convert_tokens_to_ids([label_token(name) for name in labels])


class ClauseEncoder:
    """Turns clause text into token ids for training and for prediction."""

    def __init__(self, tokenizer, label_ids: list[int], max_len: int):
        self.label_ids = list(label_ids)
        self.suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False).input_ids
        self.room = max_len - len(self.suffix_ids) - 1   # clause tokens that fit
        if self.room < 1:
            raise ValueError(f"max_len={max_len} leaves no room for the clause")
        self.tokenizer = tokenizer

    def clause_ids(self, texts: list[str]) -> list[list[int]]:
        """Full (untruncated) clause tokens, batched through the fast tokenizer."""
        return self.tokenizer(list(texts), add_special_tokens=False, verbose=False).input_ids

    def prompts(self, texts: list[str]) -> tuple[list[list[int]], list[int]]:
        """Prompt ids (clause cut to fit + suffix) and each clause's full token count."""
        full = self.clause_ids(texts)
        return [ids[:self.room] + self.suffix_ids for ids in full], [len(ids) for ids in full]

    def training_examples(self, texts: list[str], targets: list[int]) -> list[list[int]]:
        """Prompt ids followed by the label token of each target class index."""
        prompts, _ = self.prompts(texts)
        return [p + [self.label_ids[t]] for p, t in zip(prompts, targets)]


@dataclass
class LabelPositionCollator:
    """Right-pads a batch and builds labels that score only the final (label) position."""
    pad_id: int

    def __call__(self, features: list[dict]) -> dict:
        seqs = [f["input_ids"] for f in features]
        T = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), T), dtype=torch.long)
        labels = torch.full((len(seqs), T), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            n = len(s)
            input_ids[i, :n] = torch.tensor(s)
            attention_mask[i, :n] = 1
            labels[i, n - 1] = s[n - 1]   # the label token; every other position is ignored
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def label_logits(model, prompts: list[list[int]], label_ids: list[int], pad_id: int,
                 batch_size: int = 64) -> np.ndarray:
    """Scores of the k label tokens at the end of each prompt: float32 array (N, k).

    Only the k label rows of the output layer are used, so we never build the
    (N, T, V) full-vocabulary logits. Prompts are batched by length to waste
    less padding; the output is in the original order.
    """
    model.eval()
    device = next(model.parameters()).device
    label_rows = model.get_output_embeddings().weight[label_ids]            # (k, d)
    use_amp = device.type == "cuda"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = np.zeros((len(prompts), len(label_ids)), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [prompts[i] for i in idx]
        T = max(len(p) for p in batch)
        input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), T), dtype=torch.long)
        for j, p in enumerate(batch):
            input_ids[j, :len(p)] = torch.tensor(p)
            attention_mask[j, :len(p)] = 1
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        last = attention_mask.sum(dim=1) - 1
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            hidden = model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = hidden[torch.arange(len(batch), device=device), last].float()   # (B, d)
        out[idx] = (h @ label_rows.float().T).cpu().numpy()
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
