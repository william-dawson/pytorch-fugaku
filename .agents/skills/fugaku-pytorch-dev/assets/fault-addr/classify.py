"""Classify each major-fault sample of job 53270101 by the mapping its faulting address lies in.

usage: classify.py <addr.rankN.txt> <native_resources dir> <rank>

perf sample times and the phase records' start_s/end_s are both CLOCK_MONOTONIC-based on this
kernel (the first perf sample and the observer's first poll agree to <0.5 s), so samples are
also binned by training phase.
"""
import collections
import json
import re
import sys
from pathlib import Path

addr_file, src, rank = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def load_maps(path):
    out = []
    for line in path.read_text().splitlines():
        f = line.split(None, 5)
        lo, hi = (int(x, 16) for x in f[0].split('-'))
        name = f[5].strip() if len(f) > 5 else ''
        out.append((lo, hi, f[1], name))
    return out


maps = {k: load_maps(src / f'maps.{k}.rank{rank}.txt') for k in ('after', 'before')}


def classify(addr):
    for snapshot in ('after', 'before'):
        for lo, hi, perms, name in maps[snapshot]:
            if lo <= addr < hi:
                if not name:
                    label = f'[anon {perms}]'
                elif name.startswith('['):
                    label = f'{name} {perms}'
                else:
                    label = f'{name.rsplit("/", 1)[-1]} {perms}'
                return label if snapshot == 'after' else label + ' (before-only)'
    return '[unmapped in snapshots]'


phases = json.loads((src / f'timing.rank{rank}.json').read_text())['phases']


def phase_of(t):
    for p in phases:
        if p['start_s'] <= t <= p['end_s']:
            return f"{p['phase']}"
    return 'outside'


# "<tid> <time>: <addr> <addr sym> (<addr dso>) <ip> <ip sym> (<ip dso>)"
line_re = re.compile(r'^\s*(\d+)\s+([\d.]+):\s+([0-9a-f]+)\s+.*?\)\s+([0-9a-f]+)\s+(.*)$')
by_map = collections.Counter()
by_phase_map = collections.defaultdict(collections.Counter)
by_sym = collections.Counter()
unparsed = 0
pages = collections.defaultdict(collections.Counter)
for line in addr_file.read_text(errors='replace').splitlines():
    m = line_re.match(line)
    if not m:
        unparsed += 1
        continue
    tid, t, addr, ip, rest = m.groups()
    addr = int(addr, 16)
    label = classify(addr)
    ph = phase_of(float(t))
    by_map[label] += 1
    by_phase_map[ph][label] += 1
    if ph == 'backward':
        by_sym[(rest.strip()[:90], label)] += 1
        pages[label][addr >> 16] += 1

total = sum(by_map.values())
print(f'rank {rank}: {total} samples, {unparsed} unparsed lines')
print('\n== all samples by mapping')
for k, v in by_map.most_common(15):
    print(f'{v:6d} {100*v/total:5.1f}%  {k}')
for ph in ('forward', 'backward', 'step', 'outside'):
    c = by_phase_map.get(ph)
    if not c:
        continue
    n = sum(c.values())
    print(f'\n== {ph}: {n} samples')
    for k, v in c.most_common(8):
        print(f'{v:6d} {100*v/n:5.1f}%  {k}')
print('\n== backward: top (ip symbol, mapping)')
for (s, k), v in by_sym.most_common(15):
    print(f'{v:6d}  {k:40s} {s}')
print('\n== backward: distinct 64KiB pages faulted, and how often each refaults')
for k, c in sorted(pages.items(), key=lambda kv: -sum(kv[1].values()))[:5]:
    print(f'{k}: {len(c)} distinct pages, {sum(c.values())} faults, max {max(c.values())}/page')
