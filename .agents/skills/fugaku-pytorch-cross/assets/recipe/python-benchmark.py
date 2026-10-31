import json
import os
import socket
import time
import torch
import torch.distributed as dist

dist.init_process_group('mpi')
rank, size = dist.get_rank(), dist.get_world_size()
threads = int(os.environ['OMP_NUM_THREADS'])
torch.set_num_threads(threads)
n, repetitions = 4096, 3
for dtype in (torch.float32, torch.float64):
    a = torch.full((n, n), 0.5, dtype=dtype)
    b = torch.full((n, n), 0.25, dtype=dtype)
    c = torch.empty_like(a)
    torch.mm(a, b, out=c)
    dist.barrier()
    start = time.perf_counter()
    for _ in range(repetitions):
        torch.mm(a, b, out=c)
    seconds = torch.tensor([(time.perf_counter() - start) / repetitions], dtype=torch.float64)
    assert torch.all(c == n / 8), 'GEMM correctness failed'
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(json.dumps({'ranks': size, 'threads_per_rank': threads, 'n': n,
                          'dtype': str(dtype), 'max_rank_seconds': seconds.item(),
                          'aggregate_tflops': size * 2 * n**3 / seconds.item() / 1e12,
                          'correctness': 'PASS', 'host': socket.gethostname()}), flush=True)
dist.destroy_process_group()
