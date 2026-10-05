# Tool-call guardrail dataset

Labeled agent steps for training a detector that screens AI coding-agent tool calls and tool outputs.

## Files
- `processed/{train,val,test}.jsonl`: model-ready splits (field `text` is the detector input; `label` 1=harmful, 0=benign)
- `raw/gemini.jsonl`: all generated rows with Gemini's one-line rationale (never used as model input)
- `public/nl2bash.jsonl`: NL2Bash rows adapted to the schema

## Provenance
- Gemini-generated: 3000 rows (gemini-3.5-flash-lite=2675, gemini-3.5-flash=194, gemini-3.6-flash=79, gemini-3.7-flash=39, gemini-3.8-flash=13)
- NL2Bash (Lin et al., LREC 2018, GPL-3.0): 300 human-written benign description/command pairs
- Labels and categories are fixed per generation request by the generator script, not inferred by the LLM

## Processing
- canonical tool-call format; MinHash near-dup removal (Jaccard >= 0.8)
- cluster-level split (Jaccard >= 0.5) so paraphrases never straddle splits; 34 near-duplicates removed
- prior-steps balancing (35% empty for both labels); 54 benign docs given a real installer/health-check line

## Splits
| split | n | harmful | benign |
|---|---:|---:|---:|
| train | 2286 | 872 | 1414 |
| val | 497 | 192 | 305 |
| test | 483 | 182 | 301 |

## Categories (all splits)
| surface | label | category | n |
|---|---|---|---:|
| tool_call | benign | authorized_risky | 283 |
| tool_call | benign | hard_negative_lookalike | 285 |
| tool_call | benign | routine | 377 |
| tool_call | benign | routine_chained | 191 |
| tool_call | harmful | argument_poisoning | 96 |
| tool_call | harmful | destructive | 96 |
| tool_call | harmful | exfiltration | 96 |
| tool_call | harmful | multistep_laundering | 94 |
| tool_call | harmful | persistence | 96 |
| tool_call | harmful | safety_bypass | 96 |
| tool_call | harmful | supply_chain | 96 |
| tool_output | benign | benign_install_docs | 119 |
| tool_output | benign | benign_structured_response | 191 |
| tool_output | benign | benign_tool_description | 144 |
| tool_output | benign | hard_negative_imperative | 240 |
| tool_output | benign | routine | 190 |
| tool_output | harmful | api_response_injection | 144 |
| tool_output | harmful | file_injection | 144 |
| tool_output | harmful | tool_description_poisoning | 144 |
| tool_output | harmful | web_injection | 144 |

## Known limitations
- In-distribution test is near-saturated for a TF-IDF baseline (AUROC ~0.97); residual cue: reserved example.* hosts appear far more in harmful tool outputs.
- Single generator family (Gemini); no hand-authored rows yet.
