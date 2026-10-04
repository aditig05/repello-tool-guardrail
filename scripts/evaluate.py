"""Evaluate a trained detector on val/test (and any extra set, e.g. the red-team set).

Thresholds are chosen on validation only, then frozen:
  * thr_f1    - maximizes validation F1
  * thr_fpr1  - highest recall with validation FPR <= 1% (the operating point a real guardrail
                would ship with: blocking benign commands is what gets a guardrail switched off)

    python scripts/evaluate.py --model runs/qwen3-0.6b
    python scripts/evaluate.py --model runs/qwen3-0.6b --extra data/redteam/redteam.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import DATA, RESULTS, breakdown, encode, load_split, metrics, threshold_at_fpr


def load_model(path: str, device: str):
    path = Path(path)
    if (path / "adapter_config.json").exists():
        from peft import AutoPeftModelForSequenceClassification
        # float32 like training (transformers v5 otherwise defaults to the checkpoint's bf16 and
        # rounds the trained head); the "score.weight MISSING" load notice is the base model's
        # fresh head, which the adapter's saved head then replaces.
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            path, num_labels=2, dtype=torch.float32).merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    dtype = torch.float16 if device == "cuda" else torch.float32
    return model.to(device, dtype=dtype).eval(), tok


@torch.no_grad()
def predict(model, tok, rows, max_len, bs, device):
    probs, t0 = [], time.time()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))  # length-bucketed batches
    out = np.zeros(len(rows))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = encode(tok, [rows[j]["text"] for j in idx], max_len)
        batch = tok.pad(enc, return_tensors="pt").to(device)
        logits = model(**batch).logits.float()
        out[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    ms = 1000 * (time.time() - t0) / max(1, len(rows))
    return out, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--extra", nargs="*", default=[], help="extra jsonl files with a 'text' field")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap rows per set (smoke tests)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)

    sets = {s: load_split(s, Path(args.data)) for s in ("val", "test", "test_ood")}
    for p in args.extra:
        p = Path(p)
        rows = [json.loads(l) for l in p.open()]
        sets[p.stem] = rows
    sets = {k: (v[: args.limit] if args.limit else v) for k, v in sets.items() if v}
    for rows in sets.values():
        for r in rows:
            r["cell"] = f"{r['surface']}/{r['category']}"

    preds, lat = {}, {}
    for name, rows in sets.items():
        preds[name], lat[name] = predict(model, tok, rows, args.max_len, args.bs, device)

    yv, pv = np.array([r["label"] for r in sets["val"]]), preds["val"]
    prec, rec, thr = precision_recall_curve(yv, pv)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    thresholds = {"thr_0.5": 0.5, "thr_f1": float(thr[np.argmax(f1[:-1])]), "thr_fpr1": threshold_at_fpr(yv, pv, 0.01)}

    report = {"model": args.model, "device": device, "thresholds": thresholds, "latency_ms_per_sample": lat, "sets": {}}
    for name, rows in sets.items():
        y, p = [r["label"] for r in rows], preds[name]
        report["sets"][name] = {
            "overall": {t: metrics(y, p, v) for t, v in thresholds.items()},
            "by_cell@thr_f1": breakdown(rows, p, thresholds["thr_f1"], key="cell"),
            "by_source@thr_f1": breakdown(rows, p, thresholds["thr_f1"], key="source") if "source" in rows[0] else {},
        }
        m = report["sets"][name]["overall"]["thr_f1"]
        print(f"{name:10s} n={m['n']:4d}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"FPR={m['fpr']:.3f}  AUROC={m.get('auroc', float('nan')):.3f}  "
              f"TPR@1%FPR={m.get('tpr@1%fpr', float('nan')):.3f}  ({lat[name]:.1f} ms/sample)")

    RESULTS.mkdir(exist_ok=True)
    name = Path(args.model).name
    (RESULTS / f"eval_{name}.json").write_text(json.dumps(report, indent=2))
    with (RESULTS / f"preds_{name}.jsonl").open("w") as f:
        for s, rows in sets.items():
            for r, p in zip(rows, preds[s]):
                f.write(json.dumps({"set": s, "id": r.get("id"), "cell": r["cell"], "label": r["label"],
                                    "p_harmful": round(float(p), 5)}) + "\n")
    print(f"wrote results/eval_{name}.json and results/preds_{name}.jsonl")


if __name__ == "__main__":
    main()
