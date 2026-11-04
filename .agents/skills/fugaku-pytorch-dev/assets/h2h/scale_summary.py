"""Step-time breakdown per run of a TORCH_FUGAKU_COMM_STATS=2 sweep, from timing.rank<N>.json.

usage: python3 scale_summary.py results.<job>/<case> ...
Per run, mean over ranks and over the steady steps (index >= H2H_WARMUP), in seconds per step:
skew (barrier before each collective: waiting for the last rank), transfer per kind, and the
rest (compute and Python). GB is per rank per step; rss is the max-rank peak.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

KINDS = ("allgather", "reduce_scatter", "allreduce", "broadcast", "other")
rows = []
for d in map(Path, sys.argv[1:]):
    env = (d.parent / "env.txt").read_text()
    nodes = int(re.search(r"PJM_NODE=(\d+)", env).group(1))
    rpn = int(re.search(r"H2H_RPN=(\d+)", env).group(1))
    warm = int(re.search(r"H2H_WARMUP=(\d+)", env).group(1))
    wall, skew, xfer, gb, walls_max = [], [], {k: [] for k in KINDS}, {k: [] for k in KINDS}, []
    per_step_wall = {}
    for f in d.glob("timing.rank*.json"):
        t = json.loads(f.read_text())
        for s in t["steps"]:
            if s["step"] < warm or "comm" not in s:
                continue
            c = s["comm"]
            wall.append(s["wall_s"])
            per_step_wall.setdefault(s["step"], []).append(s["wall_s"])
            skew.append(sum(c[k]["skew_s"] for k in KINDS))
            for k in KINDS:
                xfer[k].append(c[k]["s"])
                gb[k].append(c[k]["bytes"] / 1e9)
    if not wall:
        print(f"{d}: no steady steps with comm stats", file=sys.stderr)
        continue
    rss = None
    log = (d / "rank0.log").read_text()
    m = re.search(r'"peak_rss_gb_max_rank": ([0-9.]+)', log)
    rss = float(m.group(1)) if m else None
    r = dict(run=str(d), nodes=nodes, layout=f"{rpn}x{48 // rpn}", ranks=nodes * rpn,
             step_s=st.mean(st.mean(v) for v in per_step_wall.values()), skew_s=st.mean(skew),
             **{f"{k}_s": st.mean(v) for k, v in xfer.items()},
             **{f"{k}_gb": st.mean(v) for k, v in gb.items()}, peak_rss_gb=rss)
    r["comm_s"] = r["skew_s"] + sum(r[f"{k}_s"] for k in KINDS)
    r["compute_s"] = r["step_s"] - r["comm_s"]
    rows.append(r)
rows.sort(key=lambda r: (r["layout"], r["nodes"]))
hdr = ["layout", "nodes", "ranks", "step_s", "compute_s", "skew_s", "allgather_s", "reduce_scatter_s",
       "allreduce_s", "other_s", "allgather_gb", "reduce_scatter_gb", "peak_rss_gb"]
print(" ".join(f"{h:>14}" for h in hdr))
for r in rows:
    print(" ".join(f"{r[h]:>14.2f}" if isinstance(r[h], float) else f"{str(r[h]):>14}" for h in hdr))
json.dump(rows, open("scale_summary.json", "w"), indent=1)
