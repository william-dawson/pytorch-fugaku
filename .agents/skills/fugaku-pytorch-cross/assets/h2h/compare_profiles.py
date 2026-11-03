"""Self CPU seconds per operator, container vs LLVM, from run.py's profile.rank<N>.txt (2 steps)."""
import re, sys
def load(path):
    rows = {}
    for line in open(path).read().split('\n\n')[0].splitlines():
        m = re.match(r'\s*(.+?)\s{2,}([\d.]+)%\s+([\d.]+)(us|ms|s)\s', line)
        if m:
            v = float(m.group(3)) * {'us': 1e-6, 'ms': 1e-3, 's': 1.0}[m.group(4)]
            calls = int(line.split()[-1])
            rows[m.group(1).strip()] = (v, calls)
    return rows
a, b = load(sys.argv[1]), load(sys.argv[2])
names = sorted(set(a) | set(b), key=lambda n: -abs(b.get(n, (0,))[0] - a.get(n, (0,))[0]))
print(f"{'operator':60s} {'container':>10s} {'llvm':>10s} {'llvm-cont':>10s}  calls c/l")
for n in names[:25]:
    x, y = a.get(n, (0, 0)), b.get(n, (0, 0))
    print(f"{n[:60]:60s} {x[0]:10.3f} {y[0]:10.3f} {y[0]-x[0]:+10.3f}  {x[1]}/{y[1]}")
print(f"{'TOTAL (listed rows)':60s} {sum(v for v, _ in a.values()):10.3f} {sum(v for v, _ in b.values()):10.3f}")
