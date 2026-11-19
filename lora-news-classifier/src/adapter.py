"""Build a classifier, and save / reload just the parts we trained.

The pretrained base is never saved: anyone can download it again. What
makes our classifier ours is small:
  * every LoRA A and B matrix,
  * the k new label-token embedding rows,
  * the Config (base model, target layers, rank, alpha, labels, max_len...),
    which says how to rebuild the frozen part before the weights go back in.

All three go into one file with torch.save. Loading = rebuild a fresh
classifier from the Config (pretrained base + LoRA layers with fresh random A,
zero B, fresh label rows), then overwrite the trainable tensors with the saved
ones. If that works, the reloaded model must give exactly the same logits.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.classify import HeadlineClassifier
from src.config import Config
from src.lora import inject_lora, lora_parameters

FORMAT_VERSION = 1


def build_classifier(cfg: Config, device: str) -> HeadlineClassifier:
    """Load the pretrained base, freeze it, add LoRA and the label tokens."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=torch.float32).to(device)
    inject_lora(model, cfg.target_modules(), cfg.rank, cfg.alpha, cfg.lora_dropout)
    return HeadlineClassifier(model, tokenizer, cfg.labels, cfg.max_len).to(device)


def adapter_state(clf: HeadlineClassifier) -> dict[str, torch.Tensor]:
    """The trainable tensors only, by name, copied to CPU."""
    state = {name: p.detach().cpu().clone() for name, p in lora_parameters(clf.model).items()}
    state["label_rows"] = clf.label_rows.detach().cpu().clone()
    return state


def save_adapter(clf: HeadlineClassifier, cfg: Config, path: Path) -> int:
    """Write config + LoRA weights + label rows to one file; return its size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": FORMAT_VERSION,
        "config": cfg.to_dict(),
        "vocab_size": clf.vocab_size,  # label token ids start here
        "state": adapter_state(clf),
    }, path)
    return path.stat().st_size


@torch.no_grad()
def load_adapter_into(clf: HeadlineClassifier, saved: dict) -> None:
    """Copy saved tensors into an already-built classifier, checking every name and shape."""
    if saved["vocab_size"] != clf.vocab_size:
        raise ValueError(f"vocab size {clf.vocab_size} != saved {saved['vocab_size']}: different base model?")
    state = saved["state"]
    targets = lora_parameters(clf.model)
    targets["label_rows"] = clf.label_rows
    missing, unexpected = set(targets) - set(state), set(state) - set(targets)
    if missing or unexpected:
        raise ValueError(f"adapter does not fit this model: missing {sorted(missing)[:3]}, "
                         f"unexpected {sorted(unexpected)[:3]}")
    for name, param in targets.items():
        if param.shape != state[name].shape:
            raise ValueError(f"{name}: shape {tuple(param.shape)} != saved {tuple(state[name].shape)}")
        param.copy_(state[name])


def load_adapter(path: Path, device: str) -> tuple[HeadlineClassifier, Config]:
    """Fresh pretrained base + saved adapter -> ready-to-use classifier."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unknown adapter format {saved['format_version']}")
    cfg = Config(**saved["config"])
    clf = build_classifier(cfg, device)
    load_adapter_into(clf, saved)
    return clf.eval(), cfg
