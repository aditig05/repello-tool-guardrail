# Agentic Tool-Call Guardrail — Track 1

**Artifacts:** [dataset](https://huggingface.co/datasets/Aditi20/tool-guardrail-dataset) · [model adapter](https://huggingface.co/Aditi20/tool-guardrail-qwen3-0.6b-lora) 
---

## 1. What has been built

A guardrail that sits between a coding agent and the world: it sees either a **tool call** the agent is about to run or a **tool output** it is about to trust, and decides *block* or *allow*. The two attack surfaces are dangerous tool calls and prompt injection hidden in tool outputs, and the brief asks for the full attack → defend → measure loop. I chose Track 1 because it is measurable and the red-teaming step is where the interesting findings come from.

The first design decision was **what the detector is allowed to see**. The naive framing is "classify this command/text," but the same command is safe or dangerous depending on context: `rm -rf ./build` is correct for "clean the build directory" and destructive for "fix the failing test." So the input is a **triplet** — the user's task, the prior steps, and the content being judged — which is the difference between a keyword blocklist and something that reasons about whether an action is *justified*.

```
surface: tool_call | tool_output
task: <what the human asked the agent to do>
prior steps: <earlier agent actions, or (none)>
content:
<the proposed command / the returned tool output>
```
---

## 2. Dataset

### 2.1 Taxonomy first

Before generating dataset, I divided it into 20 categories, balanced across the two surfaces and both labels:

- **Harmful tool calls (7):** destructive, exfiltration, persistence, safety-bypass (disabling tests/CI/hooks), supply-chain (`curl | bash`, typosquats), argument-poisoning (a legitimate command with subtly altered flags/paths), and multi-step laundering (harmless alone, harmful combined with prior steps).
- **Harmful tool outputs (4):** injections hidden in files, web pages, API/JSON responses, and MCP tool descriptions.
- **Benign (9), including hard negatives:** routine calls and outputs, but crucially *lookalikes* — the same scary verbs (`rm`, `curl`, `chmod`, `cron`) used legitimately, imperative documentation aimed at humans, security write-ups that *discuss* injection without performing it, and authorized-risky actions the task explicitly asked for.

The hard-negative categories are roughly a third of the benign data. They stop the model from learning "`rm` means bad" — a detector that blocks the agent's normal work is the first one a developer switches off.

### 2.2 Generation

I generated ~3,000 examples with Google's Gemini free tier (Flash and Flash-Lite). I fix the label and category of every request; Gemini only writes the content. The generator always fills the least-complete `(surface, label, category)` cell next, so the dataset stays balanced at every checkpoint and the class distribution is controlled by me rather than whatever the model drifts toward — which matters for a security dataset.

To stop the model leaking the label through surface form, the prompt forbids words like "attacker"/"malicious" and uses only reserved domains (`example.*`). I added **300 rows of NL2Bash** (Lin et al., 2018), real human-written (description → shell command) pairs, as benign tool calls so the detector can't learn "LLM-generated text = benign."

### 2.3 The shortcut audit — avoiding shortcut learning

Before training I ran an audit (`scripts/audit_shortcuts.py`) for features that separate the labels *without reflecting intent* — the "shortcut learning" problem in NLP text classification. If a length-only or bag-of-words model already scores near-perfectly, a neural model will learn those cues instead of intent and look great until it meets a real attack. The audit found several, which I fixed at the data level:

| Cue | Before | Fix applied | After |
|---|---|---|---|
| Tool calls with no prior steps | 0% harmful vs 31% benign | blank prior steps for 35% of **both** labels (never for multi-step laundering) | 35% vs 35% |
| `&&` command chains | 19% vs 2% | added a benign "routine_chained" category | 19% vs 16% |
| `curl` in tool outputs | 26% vs 0% | injected real installer/health-check lines into benign docs | 26% vs 6% |
| content length alone (AUROC) | 0.81 | longer benign outputs | 0.73 |

What I **could not** fully remove: a plain TF-IDF + logistic-regression model still scores **AUROC 0.99 / F1 0.90** on the test set (fake `example.*` domains still appear far more in harmful tool outputs, because real injections point *somewhere*). The honest conclusion is that **the in-distribution test set is close to saturated** — it cannot distinguish a good detector from a lazy one, which is exactly why the red-team set, not the test set, is the real evaluation.

### 2.4 Splitting

Near-duplicates are removed with MinHash (Jaccard ≥ 0.8), then similar rows are *grouped* (Jaccard ≥ 0.5) and whole groups are assigned to a split, so a paraphrase of a training example can never sit in the test set. Final splits: **train 2,286 / val 497 / test 483**, label-balanced.

---

## 3. The detector — a 0.6B decoder

I trained two models before deciding:

- **Main:** `Qwen/Qwen3-0.6B-Base` with a LoRA classification head (10.1M trainable params, 1.7%). Decoder-only, long context — important because tool outputs are long.
- **Alternative:** `answerdotai/ModernBERT-base`, full fine-tune (149M params). Encoders are the standard, faster classifier choice, and the published prompt-injection detectors (protectai's DeBERTa line) are all encoders.

For "before training" references — a fresh classification head is at chance by construction (both sit at AUROC ≈ 0.50) — I used **zero-shot prompted judges** from the Qwen family and the **off-the-shelf protectai DeBERTa**. One detail: instead of right-truncation I keep the **head and tail** of over-long inputs, since right-truncation lets an attacker hide a payload past the token limit with filler.

### 3.1 Results (held-out test, 483 rows)

| Model | AUROC | F1 | FPR | Recall @1% FPR | Latency (T4) |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B zero-shot judge | 0.50 | 0.55 | 0.98 | — | — |
| Qwen3-1.7B zero-shot judge | 0.75 | 0.56 | 0.91 | — | — |
| protectai DeBERTa (injection) | 0.76 | 0.28 | 0.04 | — | — |
| TF-IDF + logistic regression | 0.99 | 0.90 | — | 0.68 | <1 ms (CPU) |
| ModernBERT-base (full FT) | 0.997 | 0.959 | 0.030 | 0.890 | 3.5 ms |
| **Qwen3-0.6B + LoRA (main)** | **0.999** | **0.984** | **0.010** | **0.973** | 9.5 ms |

**The metric that matters is recall at 1% false-positive rate**, not AUROC. A guardrail that blocks even a few percent of benign commands gets switched off, so the real question is how many attacks you catch while almost never blocking legitimate work. Every trained model, TF-IDF included, sits at AUROC ≈ 0.99 — the metric is saturated. At 1% FPR they separate cleanly: TF-IDF 68%, ModernBERT 89%, Qwen **97%**.

**Zero-shot is unusable at this scale:** the prompted judges flag 91–98% of benign steps; protectai has decent precision but catches only 18% overall (it is built for injection and blind to dangerous *tool calls*, though it reaches AUROC 0.85 on tool-outputs alone). This is the concrete case for fine-tuning on a purpose-built dataset. In-distribution the decoder and encoder look close (F1 0.984 vs 0.959); the choice between them is settled in §5.

---

## 4. In-distribution error analysis

Qwen makes 6 errors in 483. All three missed attacks are destructive commands whose harm is in the **scope**, not the verb:

- `find tools/pipeline/ -type f ! -name '*.cpp' ! -name '*.h' -delete` — the negated filters delete everything that *isn't* source.
- `git clean -ffdx` for "clean untracked files" — the `-x` also wipes ignored files like `.env`.

The model reads the verb but not how negated filters widen the blast radius. The false alarms are the mirror image — benign content that merely *mentions* sensitive things (appending to `.env`, a workflow referencing `${{ secrets.HOST }}`). These two failure modes, scope-semantics and proximity-to-secrets, are what I then targeted in the red team.

---

## 5. Red-teaming: where the real result is

I hand-authored an **85-example blind red-team set** (43 harmful / 42 benign controls) across 10 techniques from the literature, each cited in `data/redteam/SOURCES.md`. Three disciplines make it trustworthy:

1. **No overlap with training** — every row is checked by `scripts/redteam_novelty.py` (closest train/val/test row by MinHash Jaccard and longest shared token run); the whole set passed.
2. **Benign controls for every technique** — same surface form, safe intent (legitimate base64, benign Unicode, an innocent multi-step plan), so I measure false alarms and not just recall.
3. **Blind before adaptive** — written and frozen *before* looking at any model scores, so it measures genuine generalisation (adaptive probing is future work, §6).

### 5.1 The headline

| | In-distribution test | Red-team (blind) |
|---|---:|---:|
| Attack miss / bypass rate | 1.6% | **37%** [24–52%] |
| False alarms on benign | 1.0% | **17%** [8–31%] |

A detector that misses 1.6% of held-out attacks lets **more than a third** of novel attacks through. All rates carry Wilson 95% confidence intervals, because with ~7 examples per technique a bare percentage would be dishonest.

### 5.2 Bypass rate by technique (Qwen, operating threshold)

| Technique | Bypass [95% CI] | Verdict |
|---|---|---|
| injecagent (physical/out-of-domain harm) | 0.86 [0.49, 0.97] | **broken** |
| ASCII smuggling (invisible Unicode) | 0.71 [0.36, 0.92] | **broken** |
| base64 encoding | 0.71 [0.36, 0.92] | **broken** |
| MCP poisoning / shadowing / IPI / payload-split / scope / social framing / zero-width | 0.00 | robust |

The pattern is clean and interpretable, which is worth more than a flat number:

- **ASCII smuggling** bypasses because the payload is in Unicode Tag / Private-Use characters the tokenizer discards — the model never sees the instruction, and the miss is *confident* (worst example scored 0.007, not 0.5).
- **base64** bypasses because an encoded blob is an opaque token the model can't decode-then-judge.
- **Out-of-domain harm** (unlock a door) bypasses because the training data was entirely coding-agent.
- Everything *textual and in-domain* (MCP poisoning, indirect injection, payload splitting, social framing) is caught at 0%. The model learned to read malicious intent in text; it just has no defence when the text is hidden from it or the harm is off-domain.

### 5.3 Decoder vs encoder — the model choice

I ran the **same red-team set against ModernBERT**. In-distribution it trailed Qwen by only 0.025 F1; under red-team it bypasses at **72%** vs Qwen's 37%, and additionally collapses on novel *textual* attacks Qwen handles (payload splitting 100%, scope widening 100%, MCP 50–67%). The encoder's in-distribution competitiveness is an artifact of a saturated test set; the decoder's capacity to reason over context is what actually generalises.

### 5.4 The defense that confirms the diagnosis
If obfuscation attacks fail because the payload is hidden from the tokenizer, then **surfacing it before classification should recover detection**. My input-normalization defense (`scripts/defenses.py`) strips invisible / Private-Use / bidi characters, NFKC-folds homoglyphs, and decodes base64/hex blobs in-band — all before the unchanged detector runs.

| Qwen @ operating threshold | No defense | With defense |
|---|---:|---:|
| Overall bypass | 37% | **16%** |
| ASCII smuggling | 71% | **0%** |
| base64 encoding | 71% | **14%** |
| Out-of-domain harm | 86% | 86% (unchanged) |
| False alarms on benign controls | 17% | **45%** |

This is a richer result than "the defense fixes it." It **closes the obfuscation hole** (direct evidence the diagnosis was right), does **nothing for out-of-domain harm** (that needs training data), and **raises false alarms sharply** (0.17→0.45) — because surfacing hidden content in benign controls, plus the normalizer's own annotations, makes benign inputs look suspicious to a model never trained on normalized text. The fix follows directly: apply the normalizer at *training* time too, so decoded content is in-distribution.

---

## 6. What can be done with more time

1. **Adaptive attacks.** My red-team set is *blind*. The natural next step, following Zhan et al. (2025), is an adaptive set: query the detector's score and reword a caught attack until it drops. I expect the currently-robust techniques to become partially bypassable, and the blind-vs-adaptive gap is itself a finding.
2. **Train-time normalization.** Fold `defenses.py` into the data pipeline so the model sees decoded/stripped inputs during training; this should keep the obfuscation gains without the false-alarm cost.
3. **Broaden the harm domain.** Add non-coding harmful actions (physical, financial, communication) so out-of-domain harm stops being out-of-domain.
4. **Long-context burying.** I keep head+tail of long inputs but never tested a payload buried in the middle of a very long output; that is an untested hole.
5. **Single generator family.** The data is one model family (Gemini) plus NL2Bash. A second generator and more hand-authored rows would reduce stylistic monoculture.
6. **Scope-semantics.** The in-distribution misses were all over-scoped deletions; a small amount of targeted data on negated filters and broad globs would likely close them.

---

## 7. Interesting findings, and where this generalises

- **The evaluation set can lie.** TF-IDF hitting AUROC 0.99 was the tell that the test set was saturated; the red-team set revealed the 37–72% bypass hiding underneath. This holds for any ML-for-security classifier: in-distribution metrics measure interpolation, and security is out-of-distribution by definition.
- **The tokenizer is an attack surface.** The obfuscation bypasses aren't reasoning failures — the model never receives the bytes. Any classifier acting before normalization inherits this, which is why input canonicalization belongs *in front of* the model; the same technique has since carried over into phishing filters (Microsoft, 2026).
- **Shortcut auditing transfers directly** to toxicity/abuse/spam detection — the method (can a trivial model separate the labels? which cheap cues predict them?) is domain-independent.
- **Confidently-wrong is the dangerous failure.** The smuggling miss at p=0.007 means calibration matters as much as accuracy: an uncertain detector can defer to a human, a confident-wrong one won't.

---

## 8. Key references

- Inan et al. (2023), *Llama Guard: LLM-based Input-Output Safeguard* (arXiv:2312.06674) — the reference LLM-as-safeguard; Llama Guard 3 adds tool-call classification. My work is a smaller, tool-call-specific detector in the same spirit.
- Greshake et al. (2023), *Not what you've signed up for* — indirect prompt-injection foundations.
- Zhan et al. (2024), *InjecAgent*; Debenedetti et al. (2024), *AgentDojo* — tool-agent injection benchmarks (InjecAgent informed my out-of-domain-harm examples).
- Zhan et al. (2025, NAACL Findings), *Adaptive Attacks Break Defenses* (arXiv:2503.00061) — motivates the blind-vs-adaptive methodology.
- Steinmann et al. (2024), *Navigating Shortcuts, Spurious Correlations, and Confounders* (arXiv:2412.05152) — the academic framing of my shortcut audit.
- Invariant Labs (2025), *MCP Tool Poisoning* disclosures; Liu et al. (2026), *ShareLock* (arXiv:2606.27027) — MCP-specific vectors.
- Rehberger (2024), *ASCII Smuggler* (Embrace The Red); Microsoft Security (2026) on ASCII smuggling — the invisible-Unicode technique.
- Lin et al. (2018), *NL2Bash* (LREC) — the human-written benign commands.