# Generation progress — 2026-10-04 18:05:54

**3000 / 3000** generated (100.0%), **0 left**, rate 34.1/min (last hour), ETA 0 min

Labels: harmful=1248 (42%), benign=1752 (58%) — target 50/50
Surfaces: tool_call=1536, tool_output=1464
Models: gemini-3.5-flash-lite=2675, gemini-3.5-flash=194, gemini-3.6-flash=79, gemini-3.7-flash=39, gemini-3.8-flash=13

| surface | label | category | have | target | done | rejects |
|---|---|---|---:|---:|---:|---:|
| tool_call | harmful | destructive | 96 | 96 | 100% | 0 |
| tool_call | harmful | exfiltration | 96 | 96 | 100% | 0 |
| tool_call | harmful | persistence | 96 | 96 | 100% | 0 |
| tool_call | harmful | safety_bypass | 96 | 96 | 100% | 0 |
| tool_call | harmful | supply_chain | 96 | 96 | 100% | 0 |
| tool_call | harmful | argument_poisoning | 96 | 96 | 100% | 0 |
| tool_call | harmful | multistep_laundering | 96 | 96 | 100% | 0 |
| tool_call | benign | routine | 240 | 240 | 100% | 0 |
| tool_call | benign | hard_negative_lookalike | 288 | 288 | 100% | 1 |
| tool_call | benign | authorized_risky | 144 | 144 | 100% | 0 |
| tool_output | harmful | file_injection | 144 | 144 | 100% | 2 |
| tool_output | harmful | web_injection | 144 | 144 | 100% | 0 |
| tool_output | harmful | api_response_injection | 144 | 144 | 100% | 0 |
| tool_output | harmful | tool_description_poisoning | 144 | 144 | 100% | 0 |
| tool_output | benign | routine | 192 | 192 | 100% | 2 |
| tool_output | benign | hard_negative_imperative | 240 | 240 | 100% | 0 |
| tool_output | benign | benign_tool_description | 144 | 144 | 100% | 1 |
| tool_call | benign | routine_chained | 192 | 192 | 100% | 0 |
| tool_output | benign | benign_structured_response | 192 | 192 | 100% | 1 |
| tool_output | benign | benign_install_docs | 120 | 120 | 100% | 0 |

| surface × label | have | target |
|---|---:|---:|
| tool_call / harmful | 672 | 672 |
| tool_call / benign | 864 | 864 |
| tool_output / harmful | 576 | 576 |
| tool_output / benign | 888 | 888 |
