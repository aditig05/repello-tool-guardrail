"""Fine-tune a sequence classifier (harmful=1 / benign=0) on data/processed.

Decoder (default Qwen/Qwen3-0.6B-Base): LoRA on all attention+MLP projections, the new
classification head ("score") trained fully. Encoder (e.g. answerdotai/ModernBERT-base): full
fine-tune. Records "before training" metrics (untrained head) and "after" metrics on
train/val/test so the report can show both.

    python scripts/train.py --model Qwen/Qwen3-0.6B-Base --out runs/qwen3-0.6b
    python scripts/train.py --model answerdotai/ModernBERT-base --out runs/modernbert --lr 3e-5
"""

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from common import DATA, RESULTS, encode, load_split, metrics

TA_PARAMS = inspect.signature(TrainingArguments.__init__).parameters
DECODER_HINTS = ("qwen", "llama", "gemma", "smollm", "phi", "mistral")


def make_ds(rows, tok, max_len):
    enc = encode(tok, [r["text"] for r in rows], max_len)
    return Dataset.from_dict({**enc, "labels": [r["label"] for r in rows]})


def probs_from_logits(logits):
    logits = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(logits, -1)[:, 1].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", default="runs/qwen3-0.6b")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4, help="2e-4 for LoRA, ~3e-5 for full FT encoders")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap rows per split (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()
    set_seed(args.seed)

    is_decoder = any(h in args.model.lower() for h in DECODER_HINTS)
    use_lora = is_decoder and not args.no_lora
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "benign", 1: "harmful"}, label2id={"benign": 0, "harmful": 1},
        dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    if use_lora:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()

    splits = {s: load_split(s, Path(args.data)) for s in ("train", "val", "test")}
    if args.limit:
        splits = {s: r[: args.limit] for s, r in splits.items()}
    ds = {s: make_ds(rows, tok, args.max_len) for s, rows in splits.items()}

    def compute_metrics(ev):
        return metrics(ev.label_ids, probs_from_logits(ev.predictions))

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.grad_accum, weight_decay=0.01,
        lr_scheduler_type="cosine", eval_strategy="steps", eval_steps=50, save_strategy="steps",
        save_steps=50, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="auroc",
        logging_steps=10, fp16=torch.cuda.is_available(), gradient_checkpointing=is_decoder,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # needed with LoRA (frozen inputs)
        report_to="none", seed=args.seed,
        # transformers v5 dropped warmup_ratio (warmup_steps < 1 is a ratio) and group_by_length
        **({"warmup_ratio": 0.06} if "warmup_ratio" in TA_PARAMS else {"warmup_steps": 0.06}),
        **({"group_by_length": True} if "group_by_length" in TA_PARAMS else {}))
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=4)])

    def evaluate_all(tag):
        res = {}
        for s in ("train", "val", "test"):
            out = trainer.predict(ds[s])
            res[s] = metrics(out.label_ids, probs_from_logits(out.predictions))
            print(f"[{tag}] {s}: {res[s]}")
        return res

    before = evaluate_all("before")
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    after = evaluate_all("after")

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    RESULTS.mkdir(exist_ok=True)
    name = Path(args.out).name
    (RESULTS / f"train_{name}.json").write_text(json.dumps({
        "model": args.model, "lora": use_lora, "args": vars(args), "train_seconds": round(train_secs),
        "n": {s: len(r) for s, r in splits.items()}, "before": before, "after": after,
        "log_history": trainer.state.log_history}, indent=2))
    print(f"saved model to {args.out} and results/train_{name}.json")


if __name__ == "__main__":
    main()
