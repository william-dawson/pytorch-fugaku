"""libfastdrop.so on the LLVM torch: statistics, reproducibility and speed of bernoulli_(p) and
F.dropout against stock PyTorch, in one process (stock timings first, then the library loaded).

Run on a compute node with the largepage python, 12 threads (one CMG). Prints JSON lines.
"""
import json
import os
from pathlib import Path
import sys
import time

fresh = Path('/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross-largepage/rebuild-largepage')
sys.path[:0] = [str(fresh/'wheel-install'), str(fresh/'runtime-site')]
import torch
import torch.nn.functional as F

lib = Path(__file__).resolve().parent / 'libfastdrop.so'
n = 4 * 1024 * 3072  # one GPT-125M MLP activation at micro-batch 4, seq 1024


def emit(**kw):
    print(json.dumps(kw), flush=True)


def timed(fn, reps=10):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


x = torch.empty(n)
stock_bern = timed(lambda: x.bernoulli_(0.9))
a = torch.randn(n)
stock_drop = timed(lambda: F.dropout(a, 0.1, training=True))
emit(check='stock', threads=torch.get_num_threads(), elements=n, bernoulli_s=stock_bern, dropout_s=stock_drop)

torch.ops.load_library(str(lib))
fast_bern = timed(lambda: x.bernoulli_(0.9))
fast_drop = timed(lambda: F.dropout(a, 0.1, training=True))
emit(check='fast', bernoulli_s=fast_bern, dropout_s=fast_drop,
     speedup_bernoulli=round(stock_bern / fast_bern, 1), speedup_dropout=round(stock_drop / fast_drop, 1))

failures = []
for p in (0.0, 0.1, 0.5, 0.9, 1.0):
    m = torch.empty(n).bernoulli_(p)
    values = set(torch.unique(m).tolist())
    mean = m.mean().item()
    tol = 6 * (p * (1 - p) / n) ** 0.5 + 1e-12  # 6 sigma
    ok = values <= {0.0, 1.0} and abs(mean - p) <= tol
    emit(check='mean', p=p, mean=mean, tol=tol, values=sorted(values), ok=ok)
    failures += [] if ok else [f'mean p={p}']

torch.manual_seed(123); m1 = torch.empty(n).bernoulli_(0.5)
torch.manual_seed(123); m2 = torch.empty(n).bernoulli_(0.5)
m3 = torch.empty(n).bernoulli_(0.5)
same = torch.equal(m1, m2); differ = (m2 != m3).float().mean().item()
emit(check='seed', reproducible=same, fraction_differs_next_call=differ)
failures += [] if same and 0.45 < differ < 0.55 else ['seed']

# Neighbour correlation of a p=0.5 mask (centred): should be ~0 (|r| < 6/sqrt(n)).
c = m1 - 0.5
r = (c[1:] * c[:-1]).mean().item() / 0.25
emit(check='lag1_correlation', r=r, bound=6 / n ** 0.5)
failures += [] if abs(r) < 6 / n ** 0.5 else ['correlation']

for dtype in (torch.float64, torch.bfloat16):
    m = torch.empty(1 << 20, dtype=dtype).bernoulli_(0.25)
    mean = m.float().mean().item()
    ok = abs(mean - 0.25) < 0.01 and m.dtype == dtype
    emit(check='dtype', dtype=str(dtype), mean=mean, ok=ok)
    failures += [] if ok else [f'dtype {dtype}']
nc = torch.empty(2048, 2048).t()  # non-contiguous
nc.bernoulli_(0.75)
ok = abs(nc.mean().item() - 0.75) < 0.01
emit(check='noncontiguous', mean=nc.mean().item(), ok=ok)
failures += [] if ok else ['noncontiguous']

y = F.dropout(torch.ones(n), 0.1, training=True)
vals = set(torch.unique(y).tolist())
scale = torch.tensor(1 / 0.9, dtype=torch.float32).item()  # dropout scales in float32
ok = vals <= {0.0, scale} and abs(y.mean().item() - 1.0) < 0.01
emit(check='dropout', values=sorted(vals), mean=y.mean().item(), ok=ok)
failures += [] if ok else ['dropout']

emit(check='summary', failures=failures, ok=not failures)
