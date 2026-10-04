"""Shared data loading, tokenization and metrics for training, evaluation and baselines."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"


def load_split(name: str, data_dir: Path = DATA) -> list[dict]:
    p = Path(data_dir) / f"{name}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def special_wrap(tokenizer) -> tuple[list[int], list[int]]:
    """Special tokens the tokenizer adds around a sequence (e.g. [CLS] ... [SEP]; none for Qwen).
    Found empirically so it works across tokenizer classes and transformers versions."""
    full = tokenizer("x", add_special_tokens=True)["input_ids"]
    core = tokenizer("x", add_special_tokens=False)["input_ids"]
    for i in range(len(full) - len(core) + 1):
        if full[i:i + len(core)] == core:
            return full[:i], full[i + len(core):]
    return [], []


def encode(tokenizer, texts: list[str], max_len: int) -> dict:
    """Tokenize, and if too long keep the head AND the tail (drop the middle).

    Plain right-truncation would let an attacker hide an injection after max_len tokens of filler;
    keeping both ends closes the easiest version of that hole (the red team still probes it).
    """
    out = {"input_ids": [], "attention_mask": []}
    prefix, suffix = special_wrap(tokenizer)
    budget = max_len - len(prefix) - len(suffix)
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            ids = ids[: budget // 2] + ids[-(budget - budget // 2):]
        ids = prefix + ids + suffix
        out["input_ids"].append(ids)
        out["attention_mask"].append([1] * len(ids))
    return out


def tpr_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(y_val, p_val, fpr_target):
    """Smallest threshold whose FPR on validation is <= target."""
    neg = np.sort(np.asarray(p_val)[np.asarray(y_val) == 0])
    if len(neg) == 0:
        return 0.5
    k = int(np.floor(len(neg) * (1 - fpr_target)))
    return float(neg[min(k, len(neg) - 1)]) + 1e-9


def metrics(y, p, thr=0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    m = {"n": int(len(y)), "threshold": float(thr),
         "accuracy": accuracy_score(y, pred),
         "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0),
         "f1": f1_score(y, pred, zero_division=0),
         "fpr": float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))}
    if len(set(y.tolist())) > 1:
        m.update({"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
                  "tpr@1%fpr": tpr_at_fpr(y, p, 0.01), "tpr@5%fpr": tpr_at_fpr(y, p, 0.05)})
    return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in m.items()}


def breakdown(rows, p, thr, key="category") -> dict:
    """Per-group detection rate (harmful groups) or false-positive rate (benign groups)."""
    out = {}
    for g in sorted({r[key] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r[key] == g]
        labels = {rows[i]["label"] for i in idx}
        flagged = float(np.mean([p[i] >= thr for i in idx]))
        name = "recall" if labels == {1} else "fpr" if labels == {0} else "flag_rate"
        out[g] = {"n": len(idx), name: round(flagged, 4)}
    return out
