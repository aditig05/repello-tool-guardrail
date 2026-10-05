"""Evaluate a trained detector against the red-team set.

Builds the detector input exactly like training (scripts/prepare_data.to_text + tool-call
normalization), scores every row, and applies the thresholds already frozen on validation
(results/eval_<model>.json: thr_f1 and the 1%-FPR operating point). Reports, with Wilson 95%
confidence intervals:
  * bypass rate   = harmful rows scored BELOW threshold (attacks that get through)
  * FPR           = benign controls scored at/above threshold (false alarms)
per technique, per phase (blind/adaptive), and overall.

    python scripts/redteam_eval.py --model runs/qwen3-0.6b-lora --data data/redteam/redteam.jsonl
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import RESULTS
from evaluate import load_model, predict
from prepare_data import normalize_tool_call, to_text

import torch


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.2f} [{lo:.2f},{hi:.2f}] ({k}/{n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/qwen3-0.6b-lora")
    ap.add_argument("--data", default="data/redteam/redteam.jsonl")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--defend", action="store_true", help="apply input-normalization defense before scoring")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.defend:
        from defenses import normalize_input
        for r in rows:
            r["content"] = normalize_input(r["content"])
    for r in rows:
        c = normalize_tool_call(r["content"]) if r["surface"] == "tool_call" else r["content"]
        r["text"] = to_text({"surface": r["surface"], "user_task": r["user_task"],
                             "prior_steps": r.get("prior_steps", ""), "content": c})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    p, lat = predict(model, tok, rows, args.max_len, args.bs, device)

    name = Path(args.model).name
    thr = json.load(open(RESULTS / f"eval_{name}.json"))["thresholds"]
    report = {"model": args.model, "n": len(rows), "thresholds": thr,
              "latency_ms_per_sample": round(lat, 2), "by_threshold": {}}

    for tname, tv in thr.items():
        harmful = [(r, p[i]) for i, r in enumerate(rows) if r["label"] == 1]
        benign = [(r, p[i]) for i, r in enumerate(rows) if r["label"] == 0]
        bypass = sum(s < tv for _, s in harmful)
        fp = sum(s >= tv for _, s in benign)
        block = {"overall_bypass": fmt(bypass, len(harmful)),
                 "overall_fpr_on_controls": fmt(fp, len(benign)),
                 "by_technique": {}, "by_phase": {}}
        for tech in sorted({r["technique"] for r, _ in harmful}):
            h = [(r, s) for r, s in harmful if r["technique"] == tech]
            block["by_technique"][tech] = fmt(sum(s < tv for _, s in h), len(h))
        for ph in sorted({r.get("phase", "blind") for r, _ in harmful}):
            h = [(r, s) for r, s in harmful if r.get("phase", "blind") == ph]
            block["by_phase"][ph] = fmt(sum(s < tv for _, s in h), len(h))
        report["by_threshold"][tname] = block

    print(f"model={name}  rows={len(rows)}  latency={lat:.1f} ms/sample\n")
    for tname, block in report["by_threshold"].items():
        print(f"== threshold {tname}={thr[tname]:.3f} ==")
        print(f"  bypass rate (harmful through):  {block['overall_bypass']}")
        print(f"  FPR on benign controls:        {block['overall_fpr_on_controls']}")
        for tech, v in block["by_technique"].items():
            print(f"    {tech:28s} bypass={v}")
        print()

    # per-row scores for error inspection
    detail = [{"id": r["id"], "technique": r["technique"], "label": r["label"],
               "phase": r.get("phase", "blind"), "p_harmful": round(float(p[i]), 4)}
              for i, r in enumerate(rows)]
    RESULTS.mkdir(exist_ok=True)
    suffix = "_defended" if args.defend else ""
    (RESULTS / f"redteam_{name}{suffix}.json").write_text(json.dumps({**report, "rows": detail}, indent=2))
    print(f"wrote {RESULTS / f'redteam_{name}.json'}")


if __name__ == "__main__":
    main()
