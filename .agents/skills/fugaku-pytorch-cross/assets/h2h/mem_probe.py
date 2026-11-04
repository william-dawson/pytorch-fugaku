"""Memory of one run across all ranks: per phase, the max-rank RSS at the phase end and the
max-rank peak (VmHWM); from the per-node samplers (H2H_MEMSAMPLE=1), the lowest MemAvailable on
any node and the largest per-node sum of training-process RSS.

usage: python3 mem_probe.py results.<job>/<case>
"""
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
phases, setup, steps = {}, None, {}
for f in d.glob("rank*.log"):
    for line in f.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "phase":
            k = (e["step"], e["phase"])
            p = phases.setdefault(k, [0, 0, 0])
            p[0], p[1], p[2] = max(p[0], e["rss_end_gb"]), max(p[1], e["rss_peak_gb"]), p[2] + 1
        elif e.get("event") == "setup" and setup is None:
            setup = e
        elif e.get("event") == "step" and f.name == "rank0.log":
            steps[e["step"]] = e["time_s"]
        elif e.get("event") == "mem" and e.get("step") is None:
            k = (-1, e["phase"])
            p = phases.setdefault(k, [0, 0, 0])
            p[0], p[2] = max(p[0], e["rss_gb"]), p[2] + 1
if setup:
    print({k: setup[k] for k in ("model", "params", "world", "layers", "zero_init", "loss_chunk") if k in setup})
print("step times:", steps)
for (s, ph), (end, peak, n) in sorted(phases.items()):
    print(f"  step {s:>2} {ph:<14} max rss end {end:6.2f} GB  max peak {peak:6.2f} GB  ({n} ranks)")
avail, node_sum = None, 0.0
for f in d.glob("mem.node*.txt"):
    for line in f.read_text().splitlines():
        m = re.search(r"avail_kb=(\d+)", line)
        if not m:
            continue
        a = int(m.group(1)) / 2**20
        avail = a if avail is None else min(avail, a)
        rss = [int(x) for x in line.split()[2:] if x.isdigit()]
        node_sum = max(node_sum, sum(rss) / 2**20)
print(f"nodes: min MemAvailable {avail} GB, max per-node training RSS {node_sum:.2f} GB")
