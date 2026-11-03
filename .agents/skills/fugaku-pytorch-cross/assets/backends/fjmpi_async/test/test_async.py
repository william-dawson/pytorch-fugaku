"""fjmpi_async on the LLVM torch: queued allgathers give the same bytes as the synchronous path,
in any wait order and across interleaved collectives; then a torch-level overlap timing.

Prints one JSON line per check on rank 0. Run under mpiexec with RANK/WORLD_SIZE/MASTER_* set
(rank.sh); TORCH_FUGAKU_ASYNC selects queued (1) or synchronous (0) gathers.
"""
import json
import os
from pathlib import Path
import sys
import time

fresh = Path('/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross-largepage/rebuild-largepage')
here = Path(__file__).resolve().parent
sys.path[:0] = [str(fresh/'wheel-install'), str(fresh/'runtime-site'), os.environ.get('H2H_ASYNC_PATH', str(here.parent/'backend'))]

import torch
import torch.distributed as dist
import fjmpi_async

fjmpi_async.register_backend()
dist.init_process_group('fjmpi_async')
rank, world = dist.get_rank(), dist.get_world_size()
chunk_elems = (int(os.environ.get('TORCH_FUGAKU_AG_STAGE_MB', '64')) << 20) // world // 4  # float32 per chunk


def emit(**kw):
    if rank == 0:
        print(json.dumps(kw), flush=True)


def piece(n, dtype, r, salt):
    x = torch.arange(n, dtype=torch.float64) * (r + 1) + r * 1000 + salt
    return (x % 251).to(dtype) if dtype in (torch.bfloat16, torch.int8) else x.to(dtype)


def expected(n, dtype, salt):
    return torch.cat([piece(n, dtype, r, salt) for r in range(world)])


# 1. Many gathers in flight, sizes around the chunk boundary, mixed dtypes; a blocking
#    all_reduce in the middle (drains the queue); waits in reverse order.
sizes = [0, 1, 7, 1000, chunk_elems - 1, chunk_elems, chunk_elems + 1, 2 * chunk_elems + 3, 5 * chunk_elems // 2]
cases = []
for i, n in enumerate(sizes):
    for dtype in (torch.float32, torch.bfloat16, torch.int64):
        cases.append((n, dtype, i * 10 + len(cases)))
pending = []
for k, (n, dtype, salt) in enumerate(cases):
    out = torch.full((n * world,), -1, dtype=dtype)
    h = dist.all_gather_into_tensor(out, piece(n, dtype, rank, salt), async_op=True)
    pending.append((out, h, n, dtype, salt))
    if k == len(cases) // 2:
        s = torch.tensor([float(rank)])
        dist.all_reduce(s)
        assert s.item() == world * (world - 1) / 2, s
bad = []
for out, h, n, dtype, salt in reversed(pending):
    h.wait()
    if not torch.equal(out, expected(n, dtype, salt)):
        bad.append(dict(n=n, dtype=str(dtype)))
flag = torch.tensor([len(bad)])
dist.all_reduce(flag)
emit(check='queued_vs_expected', cases=len(cases), failures_all_ranks=int(flag.item()), failures_rank0=bad[:5])

# 2. is_completed() polling alone must finish a gather (no wait()).
n = 3 * chunk_elems + 5
out = torch.empty(n * world)
h = dist.all_gather_into_tensor(out, piece(n, torch.float32, rank, 7), async_op=True)
polls = 0
t0 = time.perf_counter()
while not h.is_completed():
    polls += 1
    if time.perf_counter() - t0 > 60:
        break
ok = h.is_completed() and torch.equal(out, expected(n, torch.float32, 7))
flag = torch.tensor([0 if ok else 1])
dist.all_reduce(flag)
emit(check='is_completed_polling', polls_rank0=polls, failures_all_ranks=int(flag.item()))

# 3. Torch-level overlap: gather 8 MB per rank, either waited at once then compute, or started,
#    computed over, then waited. Compute is sized to roughly the gather time.
n = (8 << 20) // 4
src = piece(n, torch.float32, rank, 3)
out = torch.empty(n * world)
a = torch.randn(1024, 1024)


def gather_now():
    dist.all_gather_into_tensor(out, src, async_op=True).wait()


def compute(k):
    for _ in range(k):
        torch.mm(a, a)


def timed(fn, reps=10):
    ts = []
    for i in range(reps + 1):
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        t = torch.tensor([time.perf_counter() - t0])
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if i:
            ts.append(t.item())
    return sorted(ts)[len(ts) // 2]


t_comm = timed(gather_now)
t_one = timed(lambda: compute(1))
k = max(1, round(t_comm / t_one))
t_comp = timed(lambda: compute(k))


def overlapped():
    h = dist.all_gather_into_tensor(out, src, async_op=True)
    compute(k)
    h.wait()


t_serial = timed(lambda: (gather_now(), compute(k)))
fjmpi_async.reset_stats()
t_both = timed(overlapped)
st = fjmpi_async.stats()
emit(check='overlap', backend=fjmpi_async.__file__, mode='async' if st['async'] else 'sync', bytes_per_rank=n * 4, comm_s=round(t_comm, 5),
     compute_s=round(t_comp, 5), serial_s=round(t_serial, 5), overlapped_s=round(t_both, 5),
     overlap_pct=round(100 * (t_serial - t_both) / min(t_comm, t_comp), 1), mm_per_rep=k,
     correct=bool(torch.equal(out, expected(n, torch.float32, 3))), stats=st)
dist.destroy_process_group()
