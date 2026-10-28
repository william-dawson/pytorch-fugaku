"""The "fjmpi" torch.distributed backend against locally computed answers, under
mpirun/mpiexec with 2+ ranks. Every rank's input comes from a seed every rank knows,
so each rank computes the exact expected result without communicating. Sizes cross the
8 MB staging buffer; a subgroup checks communicator creation.

mpirun -np 4 python test_backend_mpi.py
"""
import os
import sys

import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMIX_RANK", "0"))))
size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29512")
import torch_fugaku  # noqa: E402

torch_fugaku.register_backend()
dist.init_process_group("fjmpi", rank=rank, world_size=size)
assert dist.get_backend() == "fjmpi", dist.get_backend()


def data(r, n, dtype, salt=0):
    g = torch.Generator().manual_seed(1000 * r + salt)
    if dtype.is_floating_point:
        return torch.randn(n, generator=g).to(dtype)
    return torch.randint(-100, 100, (n,), generator=g).to(dtype)


fails = []


def check(name, got, expect):
    exact = not got.dtype.is_floating_point
    tol = {torch.float32: 1e-5, torch.float64: 1e-12, torch.bfloat16: 2e-2}.get(got.dtype, 0)
    ok = torch.equal(got, expect) if exact else torch.allclose(got.double(), expect.double(), rtol=tol, atol=tol)
    if not ok:
        fails.append(name)
        print(f"rank {rank} FAIL {name}: max diff {(got.double() - expect.double()).abs().max().item():.3e}", flush=True)


SIZES = (1, 1000, 3_000_000)  # 3M float32 = 12 MB: more than one staging buffer (8 MB)
DTYPES = (torch.float32, torch.float64, torch.int64, torch.bfloat16)
for dtype in DTYPES:
    for n in SIZES:
        tag = f"{str(dtype)[6:]} n={n}"
        everyone = torch.stack([data(r, n, dtype) for r in range(size)])
        for op, ref in (("SUM", everyone.double().sum(0)), ("MAX", everyone.double().max(0).values),
                        ("AVG", everyone.double().mean(0))):
            if op == "AVG" and not dtype.is_floating_point:
                continue
            t = data(rank, n, dtype)
            dist.all_reduce(t, op=getattr(dist.ReduceOp, op))
            check(f"all_reduce {op} {tag}", t, ref.to(dtype))
        t = data(rank, n, dtype)
        dist.broadcast(t, src=size - 1)
        check(f"broadcast {tag}", t, everyone[size - 1])
        out = torch.empty(size * n, dtype=dtype)
        dist.all_gather_into_tensor(out, data(rank, n, dtype))
        check(f"all_gather_into_tensor {tag}", out, everyone.reshape(-1))
        outs = [torch.empty(n, dtype=dtype) for _ in range(size)]
        dist.all_gather(outs, data(rank, n, dtype))
        check(f"all_gather {tag}", torch.stack(outs), everyone)
        m = max(1, n // size)  # per-rank output of reduce_scatter / all_to_all; inputs are size * m
        big = torch.stack([data(r, size * m, dtype, salt=1) for r in range(size)])  # each rank's input
        for op in ("SUM", "AVG"):
            if op == "AVG" and not dtype.is_floating_point:
                continue
            red = big.double().sum(0) if op == "SUM" else big.double().mean(0)
            mine = red[rank * m:(rank + 1) * m].to(dtype)
            out = torch.empty(m, dtype=dtype)
            dist.reduce_scatter_tensor(out, data(rank, size * m, dtype, salt=1), op=getattr(dist.ReduceOp, op))
            check(f"reduce_scatter_tensor {op} {tag}", out, mine)
            out = torch.empty(m, dtype=dtype)
            dist.reduce_scatter(out, list(data(rank, size * m, dtype, salt=1).chunk(size)), op=getattr(dist.ReduceOp, op))
            check(f"reduce_scatter {op} {tag}", out, mine)
        t = data(rank, n, dtype)
        dist.reduce(t, dst=0)
        if rank == 0:
            check(f"reduce {tag}", t, everyone.double().sum(0).to(dtype))
        # all_to_all_single: rank r sends its chunk j to rank j.
        src = torch.stack([data(r, size * m, dtype, salt=2) for r in range(size)])
        out = torch.empty(size * m, dtype=dtype)
        dist.all_to_all_single(out, data(rank, size * m, dtype, salt=2))
        check(f"all_to_all_single {tag}", out, src[:, rank * m:(rank + 1) * m].reshape(-1))

# Uneven all_to_all: rank r sends (j + 1) rows of 3 to rank j.
rows = lambda r: sum(j + 1 for j in range(size))  # noqa: E731
mine = torch.arange(rows(rank) * 3, dtype=torch.float32).reshape(-1, 3) + 1000 * rank
out = torch.empty((rank + 1) * size, 3)
dist.all_to_all_single(out, mine, output_split_sizes=[rank + 1] * size, input_split_sizes=[j + 1 for j in range(size)])
start = sum(j + 1 for j in range(rank))
expect = torch.cat([(torch.arange(rows(r) * 3, dtype=torch.float32).reshape(-1, 3) + 1000 * r)[start:start + rank + 1]
                    for r in range(size)])
check("all_to_all_single uneven", out, expect)

# A subgroup of every rank but 0: its own communicator.
members = list(range(1, size))
sub = dist.new_group(members, backend="fjmpi")
if rank in members:
    t = torch.full((5,), float(rank))
    dist.all_reduce(t, group=sub)
    check("subgroup all_reduce", t, torch.full((5,), float(sum(members))))
dist.barrier()

flag = torch.tensor([0 if fails else 1])
dist.all_reduce(flag, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"fjmpi backend on {size} ranks: {'PASS' if flag.item() else 'FAIL'}", flush=True)
dist.destroy_process_group()
sys.exit(0 if flag.item() else 1)
