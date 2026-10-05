"""Push the dataset and the trained adapter to the Hugging Face Hub (public).

    export HF_TOKEN=...            # or in .env (gitignored)
    python scripts/push_hf.py --user Aditi20                    # both
    python scripts/push_hf.py --user Aditi20 --only dataset     # or model

Dataset repo: processed splits + raw Gemini rows + NL2Bash + DATASET_CARD.md as the card.
Model repo:   the Qwen3-0.6B LoRA adapter + a generated model card.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def dataset_card(user, repo):
    st = json.load(open(ROOT / "data/processed/stats.json"))
    body = (ROOT / "data/DATASET_CARD.md").read_text()
    n = {s: v["n"] for s, v in st["splits"].items()}
    head = ("---\nlicense: apache-2.0\ntask_categories:\n- text-classification\n"
            "language:\n- en\ntags:\n- ai-safety\n- prompt-injection\n- agent-security\n- guardrails\n"
            f"size_categories:\n- 1K<n<10K\n---\n\n")
    return head + body + f"\n\n## Splits\ntrain={n.get('train')}, val={n.get('val')}, test={n.get('test')}\n"


def model_card(user, ds_repo):
    e = json.load(open(ROOT / "results/eval_qwen3-0.6b-lora.json"))["sets"]["test"]["overall"]
    rt = json.load(open(ROOT / "results/redteam_qwen3-0.6b-lora.json"))["by_threshold"]["thr_f1"]
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
library_name: peft
pipeline_tag: text-classification
tags:
- ai-safety
- agent-security
- prompt-injection
- guardrails
- lora
---

# Tool-call guardrail — Qwen3-0.6B-Base + LoRA

Binary detector that screens a single AI coding-agent step (a proposed **tool call** or a
**tool output** the agent is about to trust) and flags it as harmful or benign. LoRA
classification head on `Qwen/Qwen3-0.6B-Base`. Trained on
[{ds_repo}](https://huggingface.co/datasets/{ds_repo}).

## Input format
```
surface: tool_call | tool_output
task: <the user's request>
prior steps: <earlier agent actions, or (none)>
content:
<the proposed command / the tool output>
```

## Results (held-out test, 483 rows)
- F1 {e['thr_f1']['f1']:.3f}, AUROC {e['thr_f1']['auroc']:.3f}
- recall {e['thr_fpr1']['recall']:.3f} at {e['thr_fpr1']['fpr']*100:.1f}% false-positive rate

## Known limitations (from red-teaming)
Overall bypass on a held-out red-team set is {rt['overall_bypass'].split()[0]} (vs 1.6% in-distribution),
concentrated in: invisible-Unicode (ASCII smuggling), base64-encoded payloads, and out-of-domain
(non-coding) harm. An input-normalization defense closes the obfuscation gap. See the repo README
for the full analysis.

## Usage
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen3-0.6B-Base"
m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, dtype=torch.float32)
m = PeftModel.from_pretrained(m, "{user}/tool-guardrail-qwen3-0.6b-lora").merge_and_unload()
tok = AutoTokenizer.from_pretrained("{user}/tool-guardrail-qwen3-0.6b-lora")
```

Trained for the Repello AI research-engineer assignment. Not production-hardened.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--only", choices=["dataset", "model"])
    ap.add_argument("--dataset-name", default="tool-guardrail-dataset")
    ap.add_argument("--model-name", default="tool-guardrail-qwen3-0.6b-lora")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    api = HfApi(token=os.environ["HF_TOKEN"])

    if args.only != "model":
        repo = f"{args.user}/{args.dataset_name}"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        (ROOT / "data/processed/README.md").write_text(dataset_card(args.user, repo))
        for f in ["train.jsonl", "val.jsonl", "test.jsonl", "stats.json", "README.md"]:
            api.upload_file(path_or_fileobj=str(ROOT / "data/processed" / f),
                            path_in_repo=f, repo_id=repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(ROOT / "data/raw/gemini.jsonl"),
                        path_in_repo="raw/gemini.jsonl", repo_id=repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(ROOT / "data/public/nl2bash.jsonl"),
                        path_in_repo="raw/nl2bash.jsonl", repo_id=repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(ROOT / "data/redteam/redteam.jsonl"),
                        path_in_repo="redteam/redteam.jsonl", repo_id=repo, repo_type="dataset")
        print(f"dataset: https://huggingface.co/datasets/{repo}")

    if args.only != "dataset":
        repo = f"{args.user}/{args.model_name}"
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        (ROOT / "runs/qwen3-0.6b-lora/README.md").write_text(model_card(args.user, f"{args.user}/{args.dataset_name}"))
        api.upload_folder(folder_path=str(ROOT / "runs/qwen3-0.6b-lora"), repo_id=repo,
                          ignore_patterns=["checkpoint-*/*", "optimizer.pt", "*.out"])
        print(f"model: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
