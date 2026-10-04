"""Shortcut audit + classical baselines.

If a length-only model or a bag-of-words model already scores near-perfectly, the dataset has
surface cues that a neural detector will exploit instead of learning intent. We report these
numbers next to the trained models as the floor any detector must beat.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/processed"
SUSPICIOUS = ["ignore", "instruction", "agent", "assistant", "override", "curl", "rm -rf", "base64",
              "token", "secret", "password", "chmod", "crontab", "--no-verify", "sudo", "| sh", "| bash"]


def load(split):
    p = D / f"{split}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def length_feats(rows):
    return np.array([[len(r["content"]), r["content"].count("\n"), len(r["user_task"]),
                      len(r.get("prior_steps") or "")] for r in rows], dtype=float)


def report(name, y, p):
    print(f"  {name:28s} F1={f1_score(y, p > 0.5):.3f}  AUROC={roc_auc_score(y, p):.3f}")


def main():
    tr, va, te = load("train"), load("val"), load("test")
    ytr = np.array([r["label"] for r in tr])
    evals = [(n, rs) for n, rs in (("val", va), ("test", te), ("test_ood", load("test_ood"))) if rs and len({r["label"] for r in rs}) > 1]

    print("Format cues (share of content starting with '{', per label):")
    for lab in (0, 1):
        rs = [r for r in tr if r["label"] == lab]
        print(f"  label={lab}: {sum(r['content'].lstrip().startswith('{') for r in rs) / max(1, len(rs)):.2f}")

    print("\nKeyword prevalence in train (harmful vs benign):")
    for kw in SUSPICIOUS:
        h = np.mean([kw in r["content"].lower() for r in tr if r["label"] == 1])
        b = np.mean([kw in r["content"].lower() for r in tr if r["label"] == 0])
        if h + b > 0.02:
            print(f"  {kw!r:16s} harmful {h:.2f}  benign {b:.2f}")

    models = {
        "length-only LR": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), length_feats),
        "TF-IDF (word+char) LR": (make_pipeline(
            make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
            LogisticRegression(max_iter=2000, C=4.0)), lambda rs: [r["text"] for r in rs]),
    }
    results = {}
    for name, (model, feat) in models.items():
        model.fit(feat(tr), ytr)
        print(f"\n{name}")
        for split, rs in evals:
            y = np.array([r["label"] for r in rs])
            p = model.predict_proba(feat(rs))[:, 1]
            report(split, y, p)
            results[f"{name}/{split}"] = {"f1": f1_score(y, p > 0.5), "auroc": roc_auc_score(y, p)}

    lr = models["TF-IDF (word+char) LR"][0]
    vocab = np.concatenate([v.get_feature_names_out() for _, v in lr.steps[0][1].transformer_list])
    coef = lr.steps[1][1].coef_[0]
    top = np.argsort(coef)
    print("\nTop harmful-indicative n-grams:", [vocab[i] for i in top[::-1][:20]])
    print("Top benign-indicative n-grams: ", [vocab[i] for i in top[:20]])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/classical_baselines.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
