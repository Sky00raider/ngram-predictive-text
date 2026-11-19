"""The classification recipe: a language model that answers with one label token.

Each headline becomes the prompt

    "<headline>\nCategory:"

and the model is trained so that the *next* token is one of k new tokens,
one per class (here <business>, <entertainment>, <sports>):

  * Label tokens.  We add k new vocabulary entries after the existing V, so
    the label ids are V, V+1, ..., V+k-1. A token needs an embedding row; in
    GPT-2 and BLOOM the output layer is tied to the input embeddings, so that
    row is also the vector the final hidden state is compared against.

    Instead of resizing the (V x d) embedding matrix we keep the k new rows in
    their own small tensor and append their scores to the model's logits.
    The maths is identical to a resized, tied embedding, but:
      - the original V rows stay frozen for free (they're not even in the
        optimizer; for BLOOM that is 257M numbers AdamW would otherwise track),
      - saving "the new rows" is just saving this one tensor.

    Each new row starts as the mean embedding of the label's English name
    (" business" etc.), a sensible starting point rather than noise.

  * Loss masked to the label position.  The prompt tokens are context only;
    the only prediction we score is the one made at the last prompt token
    (":"), whose target is the label token. Cross-entropy is over the whole
    vocabulary plus the k label tokens, as in normal language modelling.

  * Evaluation.  We take the logits at the same position and argmax over the
    k label tokens only, so the model always answers with a valid class.
"""
import torch
import torch.nn.functional as F
from torch import nn

PROMPT_SUFFIX = "\nCategory:"


class HeadlineClassifier(nn.Module):
    def __init__(self, model, tokenizer, labels: list[str], max_len: int = 64):
        super().__init__()
        self.model = model            # causal LM, already LoRA-injected (or not)
        self.tokenizer = tokenizer
        self.labels = list(labels)
        self.max_len = max_len

        embeddings = model.get_input_embeddings().weight   # (V, d), frozen
        self.vocab_size = embeddings.shape[0]               # label ids start here
        self.label_ids = list(range(self.vocab_size, self.vocab_size + len(labels)))
        self.label_rows = nn.Parameter(self._init_rows(embeddings))  # (k, d), trainable

        self.suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False).input_ids
        # GPT-2 has no padding token; any id works because padding is masked out.
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @torch.no_grad()
    def _init_rows(self, embeddings: torch.Tensor) -> torch.Tensor:
        rows = []
        for label in self.labels:
            ids = self.tokenizer(" " + label, add_special_tokens=False).input_ids
            rows.append(embeddings[ids].mean(dim=0))
        return torch.stack(rows).clone()

    # ---- inputs -----------------------------------------------------------

    def encode(self, headlines: list[str]):
        """Tokenize a batch of prompts, padded on the right.

        Returns input_ids (B, T), attention_mask (B, T) and last (B,), the
        position of each prompt's final real token, where the label is predicted.
        A headline longer than max_len keeps its beginning; the suffix is always kept.
        """
        room = self.max_len - len(self.suffix_ids)
        rows = [self.tokenizer(h, add_special_tokens=False).input_ids[:room] + self.suffix_ids
                for h in headlines]
        T = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), T), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), T), dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, :len(r)] = torch.tensor(r)
            attention_mask[i, :len(r)] = 1
        last = attention_mask.sum(dim=1) - 1
        device = self.label_rows.device
        return input_ids.to(device), attention_mask.to(device), last.to(device)

    def n_tokens(self, headline: str) -> int:
        """Tokens this headline costs (without the prompt suffix, before truncation)."""
        return len(self.tokenizer(headline, add_special_tokens=False).input_ids)

    # ---- model ------------------------------------------------------------

    def forward(self, input_ids, attention_mask, last) -> torch.Tensor:
        """Logits over the V old tokens + k label tokens, at the label position: (B, V + k)."""
        hidden = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = hidden[torch.arange(len(last), device=last.device), last]    # (B, d)
        old_logits = self.model.get_output_embeddings()(h)               # (B, V), frozen
        label_logits = h @ self.label_rows.T                             # (B, k)
        return torch.cat([old_logits, label_logits], dim=-1)

    def loss(self, headlines: list[str], targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy at the label position only. targets are class indices 0..k-1."""
        logits = self(*self.encode(headlines))
        return F.cross_entropy(logits, targets.to(logits.device) + self.vocab_size)

    @torch.no_grad()
    def label_logits(self, headlines: list[str]) -> torch.Tensor:
        """Scores for the k label tokens only: (B, k)."""
        return self(*self.encode(headlines))[:, self.vocab_size:]

    @torch.no_grad()
    def predict(self, headlines: list[str], batch_size: int = 64) -> list[int]:
        """Class index per headline, batching by length to waste less padding."""
        order = sorted(range(len(headlines)), key=lambda i: len(headlines[i]))
        preds = [0] * len(headlines)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            scores = self.label_logits([headlines[i] for i in idx])
            for i, p in zip(idx, scores.argmax(dim=-1).tolist()):
                preds[i] = p
        return preds

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
