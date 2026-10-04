"""Adapt public datasets into our schema.

nl2bash.jsonl  NL2Bash (Lin et al., LREC 2018): human-written (description, bash command) pairs.
               The description becomes user_task and the command becomes the proposed tool call, so
               every row is benign: the agent does exactly what was asked. Commands that delete, kill
               or change permissions are tagged authorized_risky, the rest routine. This adds real
               human-written benign tool calls, which keeps the detector from learning
               "Gemini-style = benign".
"""

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/public"
NL2BASH = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.{}"
RISKY = re.compile(r"\b(rm|rmdir|kill|pkill|killall|chmod|chown|shred|truncate|dd|mkfs|crontab)\b")


def fetch(url: str) -> list[str]:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8", errors="replace").splitlines()


def build_nl2bash(n: int, rng: random.Random) -> list[dict]:
    nls, cms = fetch(NL2BASH.format("nl")), fetch(NL2BASH.format("cm"))
    assert len(nls) == len(cms), (len(nls), len(cms))
    pairs = [(d.strip(), c.strip()) for d, c in zip(nls, cms) if d.strip() and c.strip()]
    # oversample risky-looking commands: they are the useful hard negatives
    risky = [p for p in pairs if RISKY.search(p[1])]
    plain = [p for p in pairs if not RISKY.search(p[1])]
    take = rng.sample(risky, min(len(risky), n // 2)) + rng.sample(plain, n - min(len(risky), n // 2))
    return [{
        "id": f"nl2bash-{i}",
        "surface": "tool_call", "label": 0,
        "category": "authorized_risky" if RISKY.search(c) else "routine",
        "user_task": d, "prior_steps": "", "content": c,
        "rationale": "Command does what the user's description asked.",
        "source": "nl2bash",
    } for i, (d, c) in enumerate(take)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl2bash", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_nl2bash(args.nl2bash, rng)
    with (OUT / "nl2bash.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cats = {c: sum(r["category"] == c for r in rows) for c in ("routine", "authorized_risky")}
    print(f"nl2bash: {len(rows)} rows {cats}")


if __name__ == "__main__":
    main()
