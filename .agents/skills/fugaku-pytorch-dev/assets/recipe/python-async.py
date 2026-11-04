"""Exercise asynchronous PyTorch MPI across two different compute nodes."""
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist

assert dist.is_mpi_available()
dist.init_process_group('mpi')
rank, size = dist.get_rank(), dist.get_world_size()
assert size == 2
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
hosts = [None] * size
dist.all_gather_object(hosts, socket.gethostname())
assert len(set(hosts)) == 2, hosts
results = []

# Warm up compute before using a delayed peer to demonstrate API non-blocking.
a = torch.ones((512, 512), dtype=torch.float64)
b = torch.full_like(a, 0.5)
def independent_compute():
    c = torch.mm(a, b)
    assert torch.all(c == 256)

independent_compute()
delay = 3.0
dist.barrier()
tensor = torch.full((1 << 20,), rank + 1, dtype=torch.float64)
if rank == 1:
    time.sleep(delay)
start = time.perf_counter()
work = dist.all_reduce(tensor, async_op=True)
api_seconds = time.perf_counter() - start
assert work is not None
pending_before_compute = not work.is_completed()
compute_start = time.perf_counter()
independent_compute()  # Does not access the communication buffer.
compute_seconds = time.perf_counter() - compute_start
pending_after_compute = not work.is_completed()
if rank == 0:
    assert api_seconds < delay / 2, api_seconds
    assert pending_before_compute and pending_after_compute
assert work.wait()
assert work.is_completed() and torch.all(tensor == 3)
results.append({'case': 'async_all_reduce_delayed_peer', 'status': 'PASS',
                'api_seconds': api_seconds, 'compute_seconds': compute_seconds,
                'pending_before_compute': pending_before_compute,
                'pending_after_compute': pending_after_compute, 'peer_delay_seconds': delay})

# Retain separate buffers and work handles for several queued collectives.
buffers = [torch.full((65536,), (rank + 1) * (i + 1), dtype=torch.float64) for i in range(3)]
works = [dist.all_reduce(buffer, async_op=True) for buffer in buffers]
futures = [handle.get_future() for handle in works]
independent_compute()
for i, (buffer, handle, future) in enumerate(zip(buffers, works, futures)):
    values = future.wait()
    assert handle.wait() and handle.is_completed() and future.done()
    assert torch.all(buffer == 3 * (i + 1))
    assert torch.equal(values[0], buffer)
results.append({'case': 'queued_all_reduce_and_futures', 'status': 'PASS', 'operations': 3})

buffer = torch.full((65536,), 7 if rank == 0 else -1, dtype=torch.int64)
work = dist.broadcast(buffer, src=0, async_op=True)
assert work.wait() and work.is_completed() and torch.all(buffer == 7)
results.append({'case': 'async_broadcast', 'status': 'PASS'})

send = torch.full((1024,), rank, dtype=torch.int64)
gathered = [torch.empty_like(send) for _ in range(size)]
work = dist.all_gather(gathered, send, async_op=True)
assert work.wait() and work.is_completed()
assert all(torch.all(value == peer) for peer, value in enumerate(gathered))
results.append({'case': 'async_all_gather', 'status': 'PASS'})

work = dist.barrier(async_op=True)
assert work.wait() and work.is_completed()
results.append({'case': 'async_barrier', 'status': 'PASS'})

# A delayed sender makes an initially pending MPI_Irecv observable.
dist.barrier()
payload = torch.arange(1 << 20, dtype=torch.int64)
receive = torch.empty_like(payload)
dist.barrier()
if rank == 0:
    time.sleep(delay)
    start = time.perf_counter()
    work = dist.isend(payload, dst=1, tag=77)
    api_seconds = time.perf_counter() - start
    independent_compute()
    assert work.wait() and work.is_completed()
    details = {'operation': 'isend', 'api_seconds': api_seconds}
else:
    start = time.perf_counter()
    work = dist.irecv(receive, src=0, tag=77)
    api_seconds = time.perf_counter() - start
    pending = not work.is_completed()
    independent_compute()
    assert api_seconds < delay / 2 and pending
    deadline = time.monotonic() + 30
    polls = 0
    while not work.is_completed():
        assert time.monotonic() < deadline, 'irecv completion polling timed out'
        polls += 1
        time.sleep(0.005)
    assert work.wait() and torch.equal(receive, payload)
    details = {'operation': 'irecv', 'api_seconds': api_seconds,
               'initially_pending': pending, 'completion_polls': polls}
results.append({'case': 'delayed_isend_irecv', 'status': 'PASS', **details})
dist.barrier()

# Prepost receive on both nodes, then send in both directions without blocking.
outgoing = torch.full((1 << 20,), 100 + rank, dtype=torch.int64)
incoming = torch.empty_like(outgoing)
recv_work = dist.irecv(incoming, src=1 - rank, tag=92)
send_work = dist.isend(outgoing, dst=1 - rank, tag=92)
independent_compute()
assert recv_work.wait() and send_work.wait()
assert recv_work.is_completed() and send_work.is_completed()
assert torch.all(incoming == 100 + (1 - rank))
results.append({'case': 'bidirectional_isend_irecv', 'status': 'PASS'})
dist.barrier()

result = {'rank': rank, 'host': socket.gethostname(), 'hosts': hosts,
          'torch': torch.__version__, 'threads': torch.get_num_threads(),
          'status': 'PASS', 'cases': results}
path = Path(__file__).with_name(f'async-results.{os.environ["PJM_JOBID"]}.rank{rank}.json')
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
dist.destroy_process_group()
