# Red-team sources log

One row per source you actually read. Fill this as you research — it becomes the citation
table in the write-up and the `reference` field of each red-team example.

| ref_id | technique family | source (title + URL) | what you took from it (mechanism, not payload) | date read |
|---|---|---|---|---|
| R01 | unicode / ASCII smuggling | Rehberger, "ASCII Smuggler" (Embrace The Red, 2024) | invisible Unicode Tags block U+E0000–U+E007F carries instructions | |
| R02 | MCP poisoning | Invariant Labs, "Tool Poisoning Attacks" (Apr 2025) | instructions hidden in tool description `<IMPORTANT>` block | |
| R03 | MCP shadowing/rug-pull | Invariant Labs, MCP security notification | description rewritten after approval; cross-tool behaviour hijack | |
| R04 | IPI foundations | Greshake et al., "Not what you've signed up for" (2023) | trusted+untrusted text concatenation in retrieved content | |
| R05 | adaptive eval | Zhan et al., arXiv:2503.00061 (NAACL Findings 2025) | why you must report blind + adaptive separately | |
| R06 | payload splitting | ShareLock, arXiv:2606.27027 | one instruction split across several tool descriptions | |
