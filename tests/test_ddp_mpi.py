"""Two or more ranks under mpirun/mpiexec: a DDP step with gradients over MPI
must give every rank the same gradients, equal to the average of the ranks'
local gradients computed without DDP (gathered over gloo). Exercises both the
explicit ``torch_fugaku.ddp()`` and the automatic hook on plain DDP.

mpirun -np 2 python test_ddp_mpi.py
"""
import os
import sys

import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMIX_RANK", "0"))))
size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")
dist.init_process_group("gloo", rank=rank, world_size=size)
import torch_fugaku  # noqa: E402


def local_grads(model, x):
    model.zero_grad()
    model(x).pow(2).sum().backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


ok = True
for mode in ("ddp()", "auto hook"):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 8))
    x = torch.randn(16, 64) + rank  # different data per rank
    mine = local_grads(model, x)
    allg = [torch.empty_like(mine) for _ in range(size)]
    dist.all_gather(allg, mine)
    expect = torch.stack(allg).mean(0)
    if mode == "ddp()":
        wrapped = torch_fugaku.ddp(model)
    else:
        torch_fugaku._hook_every_ddp()
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
    assert getattr(wrapped, "_torch_fugaku_hook", False), "hook not installed"
    model.zero_grad()
    wrapped(x).pow(2).sum().backward()
    got = torch.cat([p.grad.flatten() for p in model.parameters()])
    same = torch.allclose(got, expect, rtol=1e-5, atol=1e-6)
    ok &= same
    if rank == 0:
        print(f"   {'ok  ' if same else 'FAIL'} {mode}: MPI-averaged gradients match the gloo reference "
              f"on {size} ranks (max diff {(got - expect).abs().max().item():.1e})", flush=True)
flag = torch.tensor([int(ok)])
dist.all_reduce(flag, op=dist.ReduceOp.MIN)
if rank == 0:
    print("PASS" if flag.item() else "FAIL", flush=True)
dist.destroy_process_group()
sys.exit(0 if flag.item() else 1)
