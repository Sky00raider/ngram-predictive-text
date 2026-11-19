"""Full fine-tuning of GPT-2 on LEDGAR with the Hugging Face Trainer.

    python -m src.train --base gpt2          # all 59,080 train clauses, max_len 384 (51 min on a T4)
    python -m src.train --base distilgpt2    # 34 min; --train-size N trains on a stratified subset

Steps:
  1. Deduplicate the 60k train split and draw a stratified subset (src/data.py).
  2. Add the 100 label tokens, fix padding, resize embeddings (src/recipe.py).
  3. Train every weight with the Trainer. The loss is masked to the label
     position by the collator. After each epoch, label-position loss on a
     fixed slice of validation picks the best checkpoint.
  4. Score the *whole* validation and test splits: the 100 label-token logits
     per clause are saved, so all evaluation (accuracy, macro-F1, abstention,
     calibration, rare classes) can be done afterwards without a GPU.

Writes outputs/runs/<run>/:
  config.json        the Config of this run
  summary.json       data report, per-class train counts, timings, quick accuracy
  log_history.json   the Trainer's loss / lr / eval-loss log
  {val,test}_logits.npz   logits (N, 100), target, seen-in-train flag, clause token count
and the fine-tuned model + tokenizer to --model-dir (default <run>/model/, not in git).
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainingArguments, set_seed)

from src.config import Config, get_device
from src.data import class_counts, dedupe, load_labels, load_split, seen_mask, stratified_subset
from src.recipe import ClauseEncoder, LabelPositionCollator, add_label_tokens, label_logits

RUNS_DIR = Path("outputs/runs")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    d = Config()
    p.add_argument("--base", default=d.base, choices=["distilgpt2", "gpt2"])
    p.add_argument("--train-size", type=int, default=d.train_size)
    p.add_argument("--max-len", type=int, default=d.max_len)
    p.add_argument("--epochs", type=float, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-size", type=int, default=d.eval_size)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--run-name", default=None, help="default: <base>_n<k>k")
    p.add_argument("--model-dir", default=None, help="where the fine-tuned model goes (default: <run>/model)")
    # smoke testing the pipeline on CPU
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (smoke tests)")
    p.add_argument("--predict-limit", type=int, default=None, help="score only this many val/test clauses")
    p.add_argument("--tiny", action="store_true", help="random 2-layer model instead of pretrained weights")
    a = p.parse_args()
    cfg = Config(base=a.base, train_size=a.train_size, max_len=a.max_len, epochs=a.epochs,
                 batch_size=a.batch_size, lr=a.lr, eval_size=a.eval_size, seed=a.seed)
    return cfg, a


def load_model(cfg: Config, tiny: bool):
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_name)
    if tiny:
        model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64, n_positions=cfg.max_len))
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.hf_name)
    return model, tokenizer


def to_dataset(encoder: ClauseEncoder, rows: list[dict], index: dict[str, int]) -> Dataset:
    ids = encoder.training_examples([r["text"] for r in rows], [index[r["label"]] for r in rows])
    return Dataset.from_dict({"input_ids": ids, "length": [len(x) for x in ids]})


def score_split(model, encoder, tokenizer, rows, train_rows, index, path: Path, batch_size: int):
    prompts, n_tokens = encoder.prompts([r["text"] for r in rows])
    logits = label_logits(model, prompts, encoder.label_ids, tokenizer.pad_token_id, batch_size)
    target = np.array([index[r["label"]] for r in rows], dtype=np.int64)
    np.savez_compressed(path, logits=logits, target=target,
                        seen=np.array(seen_mask(rows, train_rows)), n_tokens=np.array(n_tokens))
    return float((logits.argmax(axis=1) == target).mean())


def main():
    cfg, args = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    run_dir = RUNS_DIR / (args.run_name or cfg.run_name())
    model_dir = Path(args.model_dir) if args.model_dir else run_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"run {run_dir}  device {device}  base {cfg.hf_name}{' (tiny random)' if args.tiny else ''}")

    # --- data ---
    labels = load_labels()
    index = {name: i for i, name in enumerate(labels)}
    train_all, dedupe_report = dedupe(load_split("train"))
    train_rows = stratified_subset(train_all, cfg.train_size, cfg.min_per_class, cfg.seed)
    val_rows, test_rows = load_split("validation"), load_split("test")
    eval_rows = random.Random(cfg.seed).sample(val_rows, min(cfg.eval_size, len(val_rows)))
    if args.predict_limit:
        val_rows, test_rows = val_rows[:args.predict_limit], test_rows[:args.predict_limit]
    counts = class_counts(train_rows, labels)
    print(f"train: {len(train_rows)} clauses (of {dedupe_report['kept']} after dedupe), "
          f"per class min {min(counts)} / median {sorted(counts)[50]} / max {max(counts)}")

    # --- model + recipe ---
    model, tokenizer = load_model(cfg, args.tiny)
    label_ids = add_label_tokens(model, tokenizer, labels)
    encoder = ClauseEncoder(tokenizer, label_ids, cfg.max_len)
    train_ds = to_dataset(encoder, train_rows, index)
    eval_ds = to_dataset(encoder, eval_rows, index)

    # --- train ---
    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        # Trainer's eval builds full-vocabulary logits in float32: 16 x 384 x 50,357 x 4 B = 1.2 GB
        # per batch already; at 32 it needed 2.5 GB and hit the T4's limit once.
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        lr_scheduler_type="linear",
        warmup_steps=cfg.warmup_frac,          # a float < 1 is a share of all steps
        weight_decay=cfg.weight_decay,
        fp16=device == "cuda",
        train_sampling_strategy="group_by_length",   # batches of similar length: less padding
        remove_unused_columns=False,                 # keep "length" for the sampler
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,                        # no optimizer state: 0.5 GB instead of 1.5 GB
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,                   # eval = label-position loss only
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=2 if device == "cuda" else 0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=LabelPositionCollator(tokenizer.pad_token_id),
                      processing_class=tokenizer)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    (model_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    (model_dir / "tagger_config.json").write_text(json.dumps({"max_len": cfg.max_len}, indent=2))
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    (run_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))

    # --- score full validation and test ---
    t0 = time.time()
    model = trainer.model
    bs = cfg.batch_size * 4
    val_acc = score_split(model, encoder, tokenizer, val_rows, train_rows, index, run_dir / "val_logits.npz", bs)
    test_acc = score_split(model, encoder, tokenizer, test_rows, train_rows, index, run_dir / "test_logits.npz", bs)
    predict_seconds = time.time() - t0

    summary = {
        "run": run_dir.name,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "dedupe": dedupe_report,
        "train_clauses": len(train_rows),
        "train_counts": counts,
        "train_seconds": round(train_seconds, 1),
        "predict_seconds": round(predict_seconds, 1),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\ntrained in {train_seconds / 60:.1f} min, scored val+test in {predict_seconds / 60:.1f} min")
    print(f"val accuracy {val_acc:.4f}   test accuracy {test_acc:.4f}")
    print(f"outputs -> {run_dir}   model -> {model_dir}")


if __name__ == "__main__":
    main()
