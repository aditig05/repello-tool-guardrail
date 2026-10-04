"""Clean, normalize, dedupe and split all data sources into train/val/test.

Sources (each row has surface/label/category/user_task/prior_steps/content):
  data/raw/gemini.jsonl         Gemini-generated (scripts/gen_gemini.py)
  data/authored/*.jsonl         hand-written by us
  data/public/*.jsonl           adapted public datasets (scripts/build_public.py)

Key decisions (see REPORT.md):
  * Tool calls are rendered in ONE canonical format, so the wrapper style (JSON vs raw shell)
    cannot leak the label.
  * Near-duplicates are removed (MinHash, Jaccard >= 0.8), then rows are grouped into
    clusters of similar samples (Jaccard >= 0.5) and whole clusters are assigned to a split,
    so paraphrases of one sample never sit on both sides of train/test.
  * Sources listed in OOD_SOURCES never enter train/val; they form a separate test_ood split.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/processed"
OOD_SOURCES = {"injecagent"}
SHELL_TOOLS = {"bash", "shell", "sh", "terminal", "run_command", "run_shell", "execute_command", "exec", "cmd"}
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def load_all() -> list[dict]:
    # Hand-adapted rows: only those marked edited=true are used; they replace their Gemini original.
    adapted = [dict(r, source="hand_adapted") for r in load_jsonl(ROOT / "data/authoring/candidates.jsonl")
               if r.get("edited")]
    replaced = {r["adapted_from"] for r in adapted}
    rows = [dict(r, source=r.get("source", "gemini")) for r in load_jsonl(ROOT / "data/raw/gemini.jsonl")
            if r["id"] not in replaced] + adapted
    for d, src in (("data/authored", "authored"), ("data/public", None)):
        for p in sorted((ROOT / d).glob("*.jsonl")):
            for r in load_jsonl(p):
                r.setdefault("source", src or p.stem)
                rows.append(r)
    return rows


def normalize_tool_call(content: str) -> str:
    """Render JSON-wrapped tool calls as plain text: shell -> the command, other tools -> `name {args}`."""
    s = content.strip()
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s
    if not isinstance(obj, dict):
        return s
    name = obj.get("tool") or obj.get("name") or obj.get("tool_name")
    args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or obj.get("input") or {}
    if not name:
        return s
    if isinstance(args, str):
        return args if str(name).lower() in SHELL_TOOLS else f"{name} {args}"
    if str(name).lower() in SHELL_TOOLS and isinstance(args, dict) and "command" in args and len(args) == 1:
        return str(args["command"])
    return f"{name} {json.dumps(args, ensure_ascii=False)}"


def to_text(r: dict) -> str:
    """The exact string the detector sees. rationale is deliberately excluded (it states the label)."""
    prior = (r.get("prior_steps") or "").strip() or "(none)"
    return (f"surface: {r['surface']}\n"
            f"task: {r['user_task'].strip()}\n"
            f"prior steps: {prior}\n"
            f"content:\n{r['content'].strip()}")


# Benign network lines as they appear in real READMEs (well-known public installers) plus the same
# shapes on reserved example.* hosts, so neither "curl" nor "example.*" stays a harmful-only cue.
DOC_NET_LINES = [
    "curl -LsSf https://astral.sh/uv/install.sh | sh",
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
    "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash",
    "curl -sSL https://install.python-poetry.org | python3 -",
    "curl -fsSL https://get.docker.com -o get-docker.sh",
    "curl -s http://localhost:8000/health",
    "curl -fsSL https://cli.example.dev/install.sh | sh",
    "wget -qO- https://releases.example.org/tool/latest.tar.gz | tar xz",
    "curl -s https://api.example.com/v1/status | jq .",
    "curl -H \"Authorization: Bearer $GITHUB_TOKEN\" https://api.github.com/repos/example/app/releases/latest",
]
AUG_RATES = {"benign_install_docs": 0.6, "hard_negative_imperative": 0.25, "routine": 0.1}


def augment_doc_network_lines(rows, rng):
    """Insert one benign installer/health-check line into a shell code block of some benign docs.

    The audit found 'curl' in 26% of harmful tool outputs and ~0% of benign ones: Gemini avoided
    network commands in benign docs even when asked. Real documentation is full of them.
    """
    n = 0
    for r in rows:
        rate = AUG_RATES.get(r["category"], 0)
        if r["surface"] != "tool_output" or r["label"] != 0 or not rate or rng.random() >= rate:
            continue
        m = re.search(r"```(?:bash|sh|shell|console)?\n", r["content"])
        if not m:
            continue
        line = rng.choice(DOC_NET_LINES)
        r["content"] = r["content"][: m.end()] + line + "\n" + r["content"][m.end():]
        r["augmented"] = "doc_network_line"
        n += 1
    return n


def balance_prior_steps(rows, target, rng):
    """Equalize the share of tool calls with no prior steps across labels.

    The audit found 'prior steps: (none)' appeared in ~31% of benign tool calls (NL2Bash has no
    session context) but 0% of harmful ones, i.e. a free label cue. We blank prior_steps on a
    random subset of each label until both reach `target`. multistep_laundering is never blanked:
    its prior steps are what make it harmful.
    """
    for lab in (0, 1):
        rs = [r for r in rows if r["surface"] == "tool_call" and r["label"] == lab]
        empty = sum(not (r.get("prior_steps") or "").strip() for r in rs)
        eligible = [r for r in rs if (r.get("prior_steps") or "").strip() and r["category"] != "multistep_laundering"]
        k = max(0, min(len(eligible), round(target * len(rs)) - empty))
        for r in rng.sample(eligible, k):
            r["prior_steps"], r["prior_blanked"] = "", True


def minhash(text: str, k: int = 5, perm: int = 128) -> MinHash:
    t = re.sub(r"\s+", " ", text.lower())
    m = MinHash(num_perm=perm)
    for i in range(max(1, len(t) - k + 1)):
        m.update(t[i:i + k].encode())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)  # fixed so the split is reproducible
    ap.add_argument("--dup", type=float, default=0.8)
    ap.add_argument("--group", type=float, default=0.5)
    ap.add_argument("--prior-empty", type=float, default=0.35, help="target share of tool calls with no prior steps, per label")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = load_all()
    n_raw = len(rows)
    for r in rows:
        if r["surface"] == "tool_call":
            r["content"] = normalize_tool_call(r["content"])
    balance_prior_steps(rows, args.prior_empty, rng)
    n_aug = augment_doc_network_lines(rows, rng)
    for r in rows:
        r["text"] = to_text(r)

    # 1) near-duplicate removal (keeps first occurrence; authored/public load after gemini but
    #    we prefer keeping them, so process them first)
    rows.sort(key=lambda r: r["source"] == "gemini")
    lsh, kept, mh = MinHashLSH(threshold=args.dup, num_perm=128), [], {}
    for i, r in enumerate(rows):
        m = minhash(r["content"])
        if lsh.query(m):
            continue
        key = f"r{i}"
        lsh.insert(key, m)
        mh[key] = m
        r["_key"] = key
        kept.append(r)
    n_dups = n_raw - len(kept)

    # 2) group similar rows (union-find over LSH neighbours at a lower threshold)
    parent = {r["_key"]: r["_key"] for r in kept}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    glsh = MinHashLSH(threshold=args.group, num_perm=128)
    for k, m in mh.items():
        for other in glsh.query(m):
            parent[find(k)] = find(other)
        glsh.insert(k, m)
    groups = defaultdict(list)
    for r in kept:
        groups[find(r["_key"])].append(r)

    # 3) assign groups to splits, greedily keeping each category near the target ratios
    ood = [r for r in kept if r["source"] in OOD_SOURCES]
    in_dist = [g for g in groups.values() if g[0]["source"] not in OOD_SOURCES]
    rng.shuffle(in_dist)
    in_dist.sort(key=len, reverse=True)  # place big clusters first
    split_cat = {s: Counter() for s in SPLITS}
    cat_total = Counter(r["category"] for g in in_dist for r in g)
    out = {s: [] for s in SPLITS}
    for g in in_dist:
        cat = Counter(r["category"] for r in g).most_common(1)[0][0]
        deficit = {s: SPLITS[s] * cat_total[cat] - split_cat[s][cat] for s in SPLITS}
        s = max(deficit, key=deficit.get)
        out[s] += g
        for r in g:
            split_cat[s][r["category"]] += 1
    if ood:
        out["test_ood"] = ood

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["id", "text", "label", "category", "surface", "source", "user_task", "prior_steps", "content", "prior_blanked", "augmented"]
    stats = {"raw": n_raw, "doc_network_lines_added": n_aug, "near_duplicates_removed": n_dups, "groups": len(groups),
             "largest_group": max(len(g) for g in groups.values()), "splits": {}}
    for s, rs in out.items():
        rng.shuffle(rs)
        with (OUT_DIR / f"{s}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False) + "\n")
        stats["splits"][s] = {
            "n": len(rs),
            "label": dict(Counter(r["label"] for r in rs)),
            "surface": dict(Counter(r["surface"] for r in rs)),
            "source": dict(Counter(r["source"] for r in rs)),
            "category": dict(sorted(Counter(r["category"] for r in rs).items())),
        }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "splits"}, indent=2))
    for s, st in stats["splits"].items():
        print(f"{s:9s} n={st['n']:5d} label={st['label']} sources={st['source']}")


if __name__ == "__main__":
    main()
