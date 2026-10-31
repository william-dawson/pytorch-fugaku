import json
import socket
import numpy as np
import torch
import torch.distributed as dist

assert dist.is_mpi_available(), 'MPI backend missing from Python bindings'
dist.init_process_group('mpi')
rank, size = dist.get_rank(), dist.get_world_size()
assert torch.backends.openmp.is_available()
torch.set_num_threads(48)
x = torch.ones((64, 64), dtype=torch.float64, requires_grad=True)
y = x @ torch.ones_like(x)
y.sum().backward()
assert torch.all(y == 64) and torch.all(x.grad == 64)
solution = torch.linalg.solve(torch.eye(3, dtype=torch.float64) * 2, torch.ones((3, 1), dtype=torch.float64) * 4)
assert torch.allclose(solution, torch.full_like(solution, 2))
array = np.arange(16, dtype=np.float64)
tensor = torch.from_numpy(array)
tensor[0] = 42
assert array[0] == 42 and np.array_equal(tensor.numpy(), array)
value = torch.full((4,), rank + 1, dtype=torch.float64)
dist.all_reduce(value)
assert torch.all(value == size * (size + 1) / 2)
hosts = [None] * size
dist.all_gather_object(hosts, socket.gethostname())
assert size >= 2 and len(set(hosts)) >= 2, hosts
print(json.dumps({'rank': rank, 'host': socket.gethostname(), 'torch': torch.__version__,
                  'module': torch.__file__, 'threads': torch.get_num_threads(),
                  'GEMM': 'PASS', 'LAPACK': 'PASS', 'autograd': 'PASS',
                  'NumPy': 'PASS', 'MPI': 'PASS', 'OpenMP': 'PASS'}), flush=True)
dist.destroy_process_group()
