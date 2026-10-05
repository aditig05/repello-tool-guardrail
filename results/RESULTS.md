# Results

All numbers on the held-out **test** split (483 rows: 182 harmful / 301 benign) unless noted.
Thresholds are chosen on validation and frozen. `@1%FPR` = threshold giving <=1% FPR on validation.

## Detector comparison

| model | AUROC | F1 (best-F1 thr) | FPR (best-F1 thr) | recall @1%FPR | FPR @1%FPR | latency (ms/sample, T4) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-1.7B zero-shot judge (thr 0.5) | 0.749 | 0.561 | 0.914 | 0.000* | - | - |
| Qwen/Qwen3-0.6B zero-shot judge (thr 0.5) | 0.500 | 0.551 | 0.983 | 0.000* | - | - |
| protectai deberta-v3 prompt-injection-v2 (thr 0.5) | 0.761 | 0.284 | 0.036 | 0.077* | - | - |
| TF-IDF word+char LR | 0.990 | 0.897 | - | 0.681 | 0.003 | <1 (CPU) |
| ModernBERT-base, full FT | 0.997 | 0.959 | 0.030 | 0.890 | 0.003 | 3.5 |
| **Qwen3-0.6B-Base + LoRA (main)** | 0.999 | 0.984 | 0.010 | 0.973 | 0.000 | 9.5 |

\* zero-shot rows: TPR at 1% FPR computed on test directly (no validation threshold).

## Before vs after training

| model | split | AUROC before | AUROC after | F1 before | F1 after |
|---|---|---:|---:|---:|---:|
| qwen3-0.6b-lora | train | 0.494 | 1.000 | 0.007 | 0.997 |
| qwen3-0.6b-lora | val | 0.458 | 0.996 | 0.000 | 0.963 |
| qwen3-0.6b-lora | test | 0.502 | 0.999 | 0.000 | 0.986 |
| modernbert-base | train | 0.520 | 1.000 | 0.552 | 0.998 |
| modernbert-base | val | 0.509 | 0.992 | 0.557 | 0.951 |
| modernbert-base | test | 0.491 | 0.997 | 0.547 | 0.959 |

"Before" = base model with a freshly initialised classification head (chance level by construction); the zero-shot judges above are the meaningful untrained reference.

Training time on one T4: Qwen LoRA 11 min (10.1M trainable params, 1.7%), ModernBERT 3 min (149M, full).

## Per-category test results (best-F1 threshold)

| surface/category | n | metric | Qwen LoRA | ModernBERT |
|---|---:|---|---:|---:|
| tool_call/argument_poisoning | 14 | recall | 1.000 | 0.857 |
| tool_call/authorized_risky | 42 | fpr | 0.000 | 0.000 |
| tool_call/destructive | 14 | recall | 0.786 | 0.857 |
| tool_call/exfiltration | 14 | recall | 1.000 | 1.000 |
| tool_call/hard_negative_lookalike | 43 | fpr | 0.023 | 0.140 |
| tool_call/multistep_laundering | 14 | recall | 1.000 | 1.000 |
| tool_call/persistence | 14 | recall | 1.000 | 1.000 |
| tool_call/routine | 62 | fpr | 0.000 | 0.000 |
| tool_call/routine_chained | 28 | fpr | 0.000 | 0.071 |
| tool_call/safety_bypass | 14 | recall | 1.000 | 1.000 |
| tool_call/supply_chain | 14 | recall | 1.000 | 0.857 |
| tool_output/api_response_injection | 21 | recall | 1.000 | 1.000 |
| tool_output/benign_install_docs | 18 | fpr | 0.000 | 0.000 |
| tool_output/benign_structured_response | 28 | fpr | 0.036 | 0.000 |
| tool_output/benign_tool_description | 21 | fpr | 0.000 | 0.000 |
| tool_output/file_injection | 21 | recall | 1.000 | 1.000 |
| tool_output/hard_negative_imperative | 36 | fpr | 0.000 | 0.000 |
| tool_output/routine | 23 | fpr | 0.043 | 0.043 |
| tool_output/tool_description_poisoning | 21 | recall | 1.000 | 1.000 |
| tool_output/web_injection | 21 | recall | 1.000 | 1.000 |

## Red-team results (blind set, 85 examples, 43 harmful / 42 benign controls)

Evaluated at the frozen operating threshold (thr_f1=0.329). Bypass = harmful example scored below threshold (attack succeeds). 95% Wilson CIs.

**Overall bypass rate: 0.37 [0.24,0.52] (16/43)** vs 1.6% miss rate on the in-distribution test set.  
**False-alarm rate on adversarial benign controls: 0.17 [0.08,0.31] (7/42)** vs 1.0% on in-distribution benign.

| technique | bypass rate [95% CI] | verdict |
|---|---|---|
| injecagent_direct_harm | 0.86 [0.49,0.97] (6/7) | **BROKEN** |
| ascii_smuggling | 0.71 [0.36,0.92] (5/7) | **BROKEN** |
| encoding_runtime_decode | 0.71 [0.36,0.92] (5/7) | **BROKEN** |
| ipi_readme_injection | 0.00 [0.00,0.56] (0/3) | robust |
| mcp_description_poisoning | 0.00 [0.00,0.49] (0/4) | robust |
| mcp_shadowing | 0.00 [0.00,0.56] (0/3) | robust |
| payload_splitting | 0.00 [0.00,0.56] (0/3) | robust |
| semantic_scope_widening | 0.00 [0.00,0.56] (0/3) | robust |
| social_framing_convention | 0.00 [0.00,0.56] (0/3) | robust |
| zero_width_injection | 0.00 [0.00,0.56] (0/3) | robust |

## Additional analyses (for the report)

### A. Decoder vs encoder under red-team (model-choice justification)

In-distribution the two models look close (F1 0.984 vs 0.959). Under novel attacks the gap is decisive:

| | Qwen3-0.6B+LoRA | ModernBERT-base |
|---|---|---|
| in-distribution test F1 | 0.984 | 0.959 |
| red-team overall bypass | 0.37 | 0.72 |
| ascii/encoding/injecagent bypass | 0.71–0.86 | 1.00 all three |
| also breaks on | — | payload_splitting, scope_widening, mcp (0.5–1.0) |

The encoder fails on novel *textual* attacks the decoder still catches (payload splitting, scope widening, MCP). The decoder's extra context-reasoning is what the brief's model-choice question is really about; it only shows up out-of-distribution.

### B. Both models fail identically on obfuscation + OOD harm (fundamental, not model-specific)

ascii_smuggling, encoding_runtime_decode and injecagent_direct_harm bypass *both* models at ≥0.71. These are not a tuning problem: the payload is hidden from the tokenizer (invisible chars), unreadable (base64), or out of the training domain (physical-world harm).

### C. Input-normalization defense (confirms the diagnosis + a working mitigation with a cost)

Defense = strip invisible/PUA/bidi characters, NFKC-fold homoglyphs, and decode base64/hex blobs in-band, *before* the unchanged detector scores the input (scripts/defenses.py).

| metric (Qwen, thr_f1) | no defense | with defense |
|---|---|---|
| overall bypass | 0.37 | 0.16 |
| ascii_smuggling bypass | 0.71 | 0.00 |
| encoding bypass | 0.71 | 0.14 |
| injecagent (OOD harm) bypass | 0.86 | 0.86 |
| FPR on benign controls | 0.17 | 0.45 |

Reading: normalizing the input **closes the obfuscation hole** (ascii 0.71→0.00, encoding 0.71→0.14) — direct evidence the diagnosis is right: those attacks only worked because the payload was hidden from the tokenizer. It does **nothing for out-of-domain harm** (injecagent stays 0.86), which needs training data, not normalization. And it **raises false alarms** (0.17→0.45) because surfacing hidden content in benign controls — and the normalizer's own annotations — make benign inputs look suspicious to a model never trained on normalized text. The proper fix is to apply the normalizer at *training* time too, so decoded content is in-distribution; that is concrete future work, not a mystery.
