"""Progress report for the Gemini generation run.

Writes data/raw/progress.md (full table) and prints a compact summary to stdout.
Keeps a small history file so it can estimate generation rate and ETA.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_gemini import CELLS, OUT, REJECTS, ROOT, cell_key, load_jsonl, targets  # noqa: E402

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
HIST = ROOT / "data/raw/.progress_hist.json"
REPORT = ROOT / "data/raw/progress.md"


def main():
    rows = load_jsonl(OUT)
    want = targets(TOTAL)
    target_total = sum(want.values())
    have = Counter(cell_key(r["surface"], r["label"], r["category"]) for r in rows)
    rej = Counter(r["cell"] for r in load_jsonl(REJECTS))
    by_label = Counter(r["label"] for r in rows)
    by_surface = Counter(r["surface"] for r in rows)
    by_sl = Counter((r["surface"], r["label"]) for r in rows)
    models = Counter(r.get("model", "?") for r in rows)
    n = len(rows)

    hist = json.loads(HIST.read_text()) if HIST.exists() else []
    now = time.time()
    hist = [h for h in hist if now - h[0] < 3600] + [[now, n]]
    HIST.write_text(json.dumps(hist))
    rate = (hist[-1][1] - hist[0][1]) / ((hist[-1][0] - hist[0][0]) / 60) if len(hist) > 1 and hist[-1][0] > hist[0][0] else 0
    left = max(0, target_total - n)
    eta = f"{left / rate:.0f} min" if rate > 0 else "n/a"
    harm_pct = 100 * by_label[1] / n if n else 0

    lines = [
        f"# Generation progress — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**{n} / {target_total}** generated ({100 * n / target_total:.1f}%), **{left} left**, "
        f"rate {rate:.1f}/min (last hour), ETA {eta}",
        "",
        f"Labels: harmful={by_label[1]} ({harm_pct:.0f}%), benign={by_label[0]} ({100 - harm_pct if n else 0:.0f}%) — target 50/50",
        f"Surfaces: tool_call={by_surface['tool_call']}, tool_output={by_surface['tool_output']}",
        f"Models: {', '.join(f'{m}={c}' for m, c in models.most_common())}",
        "",
        "| surface | label | category | have | target | done | rejects |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for s, l, c, *_ in CELLS:
        k = cell_key(s, l, c)
        lines.append(f"| {s} | {'harmful' if l else 'benign'} | {c} | {have[k]} | {want[k]} | "
                     f"{100 * have[k] / want[k]:.0f}% | {rej[k]} |")
    lines += ["", "| surface × label | have | target |", "|---|---:|---:|"]
    for s in ("tool_call", "tool_output"):
        for l in (1, 0):
            t = sum(want[cell_key(*c[:3])] for c in CELLS if c[0] == s and c[1] == l)
            lines.append(f"| {s} / {'harmful' if l else 'benign'} | {by_sl[(s, l)]} | {t} |")
    REPORT.write_text("\n".join(lines) + "\n")

    done_cells = sum(have[cell_key(*c[:3])] >= want[cell_key(*c[:3])] for c in CELLS)
    print(f"[{time.strftime('%H:%M')}] {n}/{target_total} ({100 * n / target_total:.0f}%), {left} left, "
          f"{rate:.1f}/min, ETA {eta} | harmful {by_label[1]} / benign {by_label[0]} | "
          f"call {by_surface['tool_call']} / output {by_surface['tool_output']} | "
          f"categories complete {done_cells}/{len(CELLS)} | rejects {sum(rej.values())}")
    print(f"  harmful(+) {by_label[1]}/{sum(v for k, v in want.items() if k.split('|')[1] == '1')}  "
          f"benign(-) {by_label[0]}/{sum(v for k, v in want.items() if k.split('|')[1] == '0')}  "
          f"balance {harm_pct:.0f}/{100 - harm_pct if n else 0:.0f}")
    for s_, l_, c_, *_ in CELLS:
        k = cell_key(s_, l_, c_)
        print(f"  {s_:11s} {'+' if l_ else '-'} {c_:28s} {have[k]:4d}/{want[k]:<4d} ({want[k] - have[k]} left)")


if __name__ == "__main__":
    main()
