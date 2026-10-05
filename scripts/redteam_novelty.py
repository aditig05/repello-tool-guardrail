"""Check red-team examples for schema validity and non-overlap with the training data.

Write batches as data/redteam/*.jsonl (same schema as processed data + technique/base_category/
phase/reference), then:

    python scripts/redteam_novelty.py                    # all batches
    python scripts/redteam_novelty.py data/redteam/batch_01.jsonl

For every row it reports the single closest train/val/test sample by character-5gram Jaccard
(MinHash estimate, verified exactly on the top candidate) and the longest shared run of word
tokens. Rows at/over the thresholds are flagged as possible overlap and must be reworded.
It also validates required fields, flags internal duplicates, and prints coverage per technique.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parent.parent
RT_DIR = ROOT / "data/redteam"
PROC = ROOT / "data/processed"
JACCARD_FLAG = 0.5     # char-5gram Jaccard at/above this = too similar to a training row
RUN_FLAG = 12          # shared run of this many word tokens = likely copied span
REQUIRED = {"surface", "label", "content", "user_task", "technique", "base_category", "phase"}
VALID_SURFACE = {"tool_call", "tool_output"}
VALID_PHASE = {"blind", "adaptive"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def char_shingles(s: str, k: int = 5) -> set[str]:
    t = norm(s)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def minhash(sh: set[str], perm: int = 128) -> MinHash:
    m = MinHash(num_perm=perm)
    for s in sh:
        m.update(s.encode())
    return m


def longest_token_run(a: str, b: str) -> int:
    """Longest run of consecutive word tokens shared between a and b (classic DP, token-level)."""
    ta, tb = norm(a).split(), norm(b).split()
    if not ta or not tb:
        return 0
    prev = [0] * (len(tb) + 1)
    best = 0
    for i in range(1, len(ta) + 1):
        cur = [0] * (len(tb) + 1)
        for j in range(1, len(tb) + 1):
            if ta[i - 1] == tb[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def load_train():
    rows = []
    for split in ("train", "val", "test"):
        p = PROC / f"{split}.jsonl"
        if p.exists():
            for l in p.open():
                r = json.loads(l)
                r["_split"] = split
                rows.append(r)
    return rows


def main():
    args = [Path(a) for a in sys.argv[1:]] or sorted(RT_DIR.glob("batch_*.jsonl"))
    if not args:
        print("No red-team batches found in data/redteam/*.jsonl")
        return
    train = load_train()
    if not train:
        print("No processed training data — run scripts/prepare_data.py first.")
        return
    train_sh = [char_shingles(r["content"]) for r in train]
    lsh = MinHashLSH(threshold=0.3, num_perm=128)  # loose recall; exact-verify the hits
    mh = []
    for i, sh in enumerate(train_sh):
        m = minhash(sh)
        mh.append(m)
        lsh.insert(str(i), m)

    rt_rows, seen, problems = [], {}, []
    tech = Counter()
    phase = Counter()
    label = Counter()
    for path in args:
        for ln, line in enumerate(path.open(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{path.name}:{ln} invalid JSON: {e}")
                continue
            miss = REQUIRED - set(r)
            if miss:
                problems.append(f"{path.name}:{ln} missing fields {sorted(miss)}")
                continue
            if r["surface"] not in VALID_SURFACE:
                problems.append(f"{path.name}:{ln} bad surface {r['surface']!r}")
            if r["phase"] not in VALID_PHASE:
                problems.append(f"{path.name}:{ln} bad phase {r['phase']!r}")
            if r["label"] not in (0, 1):
                problems.append(f"{path.name}:{ln} bad label {r['label']!r}")
            h = norm(r["content"])
            if h in seen:
                problems.append(f"{path.name}:{ln} duplicate of {seen[h]}")
                continue
            seen[h] = f"{path.name}:{ln}"
            r["_src"] = f"{path.name}:{ln}"
            rt_rows.append(r)
            tech[r["technique"]] += 1
            phase[r["phase"]] += 1
            label[r["label"]] += 1

    flagged = []
    print(f"\n{'red-team row':28s} {'tech':16s} {'lbl':3s} {'maxJacc':>7s} {'run':>4s}  nearest train row")
    print("-" * 110)
    for r in rt_rows:
        sh = char_shingles(r["content"])
        cand = {int(k) for k in lsh.query(minhash(sh))}
        cand |= set(range(len(train))) if not cand else set()  # fall back to full scan if no LSH hit
        best_i, best_j = -1, 0.0
        for i in cand:
            j = jaccard(sh, train_sh[i])
            if j > best_j:
                best_j, best_i = j, i
        run = longest_token_run(r["content"], train[best_i]["content"]) if best_i >= 0 else 0
        hit = best_j >= JACCARD_FLAG or run >= RUN_FLAG
        if hit:
            flagged.append((r["_src"], round(best_j, 3), run))
        mark = "  <-- REVIEW" if hit else ""
        near = f"{train[best_i]['_split']}:{train[best_i].get('category','?')}" if best_i >= 0 else "-"
        print(f"{r['_src']:28s} {r['technique'][:16]:16s} {r['label']:<3d} {best_j:7.3f} {run:4d}  {near}{mark}")

    print("\n=== summary ===")
    print(f"rows: {len(rt_rows)}  | labels: harmful={label[1]} benign={label[0]}  | phases: {dict(phase)}")
    print(f"techniques ({len(tech)}): " + ", ".join(f"{k}={v}" for k, v in sorted(tech.items())))
    if problems:
        print(f"\n{len(problems)} schema/dup problems:")
        for p in problems[:40]:
            print("  " + p)
    if flagged:
        print(f"\n{len(flagged)} rows too close to training data (reword these): "
              f"Jaccard>={JACCARD_FLAG} or shared run>={RUN_FLAG} tokens")
        for s, j, run in flagged:
            print(f"  {s}  jaccard={j} run={run}")
    else:
        print("\nNo overlap flags: every red-team row is novel vs train/val/test. ✓")


if __name__ == "__main__":
    main()
