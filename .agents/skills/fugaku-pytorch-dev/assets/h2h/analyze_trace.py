"""Why does (or doesn't) fjmpi_async overlap? Reads one rank's trace.rank<N>.json (run.py,
H2H_TRACE_STEPS) and reports, per phase (forward / backward / step) of the traced steps:

  slack   issue -> wait_begin: time the caller computed while the gather could move
  queued  issue -> first transfer start: time it sat behind earlier gathers
  blocked wait_begin -> wait_end
  pumps   MPI_Test misses between its first start and its completion: how often the backend
          was entered while the transfer was moving
plus drains (queue finished early because another collective came) and synchronous
collectives by kind.

usage: analyze_trace.py trace.rankN.json
"""
import collections
import json
import statistics
import sys

d = json.loads(open(sys.argv[1]).read())
K = {k: i for i, k in enumerate(d['kinds'])}
ev = sorted(d['events'])
phases = [p for p in d['phases'] if 'start_s' in p]


def phase_of(t):
    for p in phases:
        if p['start_s'] <= t <= p['start_s'] + p['wall_s']:
            return f"{p['phase']}"
    return 'between'


g = collections.defaultdict(dict)
misses = collections.Counter()
for t, kind, gid, b in ev:
    x = g[gid] if gid >= 0 else None
    if kind == K['issue']:
        x.update(issue=t, bytes=b)
    elif kind == K['start']:
        x.setdefault('starts', []).append(t)
    elif kind == K['done']:
        x.setdefault('dones', []).append(t)
    elif kind == K['test_miss']:
        misses[gid] += 1
    elif kind == K['wait_begin']:
        x.update(wait_begin=t, already_done=bool(b))
    elif kind == K['wait_end']:
        x['wait_end'] = t

rows = collections.defaultdict(list)
for gid, x in g.items():
    if 'issue' not in x or 'wait_begin' not in x:
        continue
    rows[phase_of(x['issue'])].append(dict(
        slack=x['wait_begin'] - x['issue'],
        queued=(x['starts'][0] - x['issue']) if x.get('starts') else 0.0,
        blocked=x['wait_end'] - x['wait_begin'],
        transfers=len(x.get('starts', [])),
        pumps=misses[gid],
        mb=x['bytes'] / 2**20,
        done_before_wait=x['already_done']))

steps = sorted(int(k) for k in d['steps'])
span = [p for p in phases if p['step'] in steps]
print(f"{sys.argv[1]}: traced steps {steps}; phases " +
      ', '.join(f"{p['step']}:{p['phase']}={p['wall_s']:.2f}s" for p in span))
for ph, rs in rows.items():
    n = len(rs)
    tot = lambda k: sum(r[k] for r in rs)
    med = lambda k: statistics.median(r[k] for r in rs)
    print(f"\n== {ph}: {n} gathers, {tot('mb'):.0f} MiB/rank in; "
          f"done before wait {sum(r['done_before_wait'] for r in rs)}/{n}")
    print(f"   slack   total {tot('slack'):7.3f}s  median {med('slack')*1e3:7.2f}ms  "
          f"(waited < 1 ms after issue: {sum(r['slack'] < 1e-3 for r in rs)})")
    print(f"   queued  total {tot('queued'):7.3f}s  median {med('queued')*1e3:7.2f}ms")
    print(f"   blocked total {tot('blocked'):7.3f}s  median {med('blocked')*1e3:7.2f}ms")
    print(f"   transfers/gather median {med('transfers')}  max {max(r['transfers'] for r in rs)};  "
          f"pumps while moving: median {med('pumps')}, zero for {sum(r['pumps'] == 0 for r in rs)}/{n}")

colls = collections.defaultdict(list)
open_ = {}
for t, kind, gid, b in ev:
    if kind in (K['coll_begin'], K['drain_begin']):
        open_[(kind, b)] = t
    elif kind in (K['coll_end'], K['drain_end']):
        start = open_.pop((kind - 1, b), None)
        if start is not None:
            name = 'drain' if kind == K['drain_end'] else {1: 'reduce_scatter', 2: 'allreduce', 3: 'other', 4: 'allgather_sync'}[b]
            colls[(phase_of(start), name)].append(t - start)
print('\n== blocking calls (phase, kind): count, total s')
for (ph, name), ts in sorted(colls.items()):
    print(f"   {ph:9s} {name:15s} {len(ts):5d} {sum(ts):8.3f}s")
