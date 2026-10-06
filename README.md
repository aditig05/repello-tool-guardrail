# Agentic Tool-Call Guardrail (Track 1)

A binary detector that screens a single AI coding-agent step, a proposed **tool call** or a
**tool output** it is about to trust and flags it *harmful* or *benign*, judged against the
user's task and prior steps. Full methodology, results, and analysis are in **[REPORT.md](REPORT.md)**.

**Artifacts (public):**
[dataset](https://huggingface.co/datasets/Aditi20/tool-guardrail-dataset) ·
[model adapter](https://huggingface.co/Aditi20/tool-guardrail-qwen3-0.6b-lora)

## Headline results (held-out test, 483 rows)

| Model | AUROC | F1 | Recall @1% FPR |
|---|---:|---:|---:|
| Qwen3-0.6B-Base + LoRA (main) | 0.999 | 0.984 | 0.973 |
| ModernBERT-base (alternative) | 0.997 | 0.959 | 0.890 |

On a manually extracted, no-overlap **red-team set**, bypass rises from 1.6% (in-distribution) to **37%**,
concentrated in invisible-Unicode, base64, and out-of-domain harm. An input-normalization defense
(`scripts/defenses.py`) closes the obfuscation gap. See REPORT.md §5.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Gemini (data generation) and Hugging Face (upload) keys — only needed for those steps:
cp .env.example .env    # then set GEMINI_API_KEY and HF_TOKEN
```

## Pipeline

Training/eval is designed for a free Kaggle T4; everything else runs on a laptop CPU.

```bash
# 1. Data: generate, adapt public data, split, audit for shortcuts
python scripts/gen_gemini.py --total 3000        # Gemini; resumable, writes data/raw/
python scripts/build_public.py                   # 300 NL2Bash benign tool calls
python scripts/prepare_data.py                   # dedup + grouped split -> data/processed/
python scripts/audit_shortcuts.py                # shortcut audit + classical baselines

# 2. Train + evaluate (on Kaggle: upload kaggle/guardrail-bundle.zip, run kaggle/train_kaggle.ipynb)
python scripts/train.py    --model Qwen/Qwen3-0.6B-Base   --out runs/qwen3-0.6b-lora
python scripts/train.py    --model answerdotai/ModernBERT-base --out runs/modernbert-base --lr 3e-5
python scripts/evaluate.py --model runs/qwen3-0.6b-lora
python scripts/baselines_zero_shot.py --judge Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --protectai

# 3. Red-team: validate novelty, evaluate, test the defense
./check_redteam.sh                                                  # novelty vs training data
python scripts/redteam_eval.py --model runs/qwen3-0.6b-lora         # per-technique bypass + CIs
python scripts/redteam_eval.py --model runs/qwen3-0.6b-lora --defend
```

`kaggle/make_bundle.sh` repackages code + processed data for Kaggle; `scripts/push_hf.py --user <you>`
publishes the dataset and adapter.

## Layout

```
scripts/     gen_gemini, build_public, prepare_data, audit_shortcuts, train, evaluate,
             baselines_zero_shot, redteam_novelty, redteam_eval, defenses, push_hf
data/        raw/ (generated)  public/ (NL2Bash)  processed/ (splits)  redteam/ (hand-authored)
results/     metrics JSON + RESULTS.md (all tables)
kaggle/      train_kaggle.ipynb, make_bundle.sh
REPORT.md    full write-up
```
