# Compute-node validation and MPI semantics

## Runtime and scheduler settings

The successful jobs used project `ra000009`, `f-pt`, and exactly one cache
declaration `PJM_LLIO_GFSCACHE=/vol0004`, with the working directory under
the project data area. `f-pt` consumes Fugaku Points: use existing session
authorization or obtain it before submitting, and check the project balance.
Do not assume another account has the same queue or power permissions.

```sh
#PJM -L rscgrp=f-pt
#PJM -L node=2
#PJM -L elapse=00:05:00
#PJM -g ra000009
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM --mpi max-proc-per-node=1
#PJM -L jobenv=linux2,freq=2000,eco_state=0
```

The account rejected `freq=2200,eco_state=0` as unauthorized boost mode.
`freq=2000,eco_state=0` worked. These are **requested** settings; the benchmark
did not independently measure clock frequency. Boost-eco results should not be
compared as if they were normal-mode measurements. Do not declare two storage
volumes as a comma-separated value: on this site that queued successfully but
later failed the pre-execution gate check with no program output.

For normal Python runs, the supplied wrapper unsets `LD_PRELOAD`/`PYTHONHOME`,
sets `PYTHONNOUSERSITE=1`, imports the installed wheel and staged dependencies,
sets its library directory plus Fujitsu's `lib64`, disables OpenMP dynamic
thread counts and limits NumPy's OpenBLAS to one thread. PyTorch's matrix
multiply uses the separately linked Fujitsu BLAS. Do not propagate unrelated
GCC cross `lib64` directories: one such runtime required a newer glibc and
prevented compute-node `mpiexec` from starting.

## Acceptance checks

| Phase | Required evidence |
|---|---|
| Core configuration | MPI/distributed/OpenMP ON; actual LAPACK link; MPI backend compilation; correct generated warning flags |
| Core artifacts | AArch64 `libc10`, `libtorch`, `libtorch_cpu`; Fujitsu BLAS dependency; exported MPI factory; expected OpenMP runtime |
| Two-node C++ smoke | Different hosts; GEMM, LAPACK solve, autograd, ProcessGroupMPI all-reduce and actual OpenMP thread-team size PASS |
| Python configuration | BUILD_PYTHON/BUILD_LIBTORCHLESS/NumPy/MPI ON; target headers; MPI binding define; no core recompilation |
| Installed-wheel smoke | Imports the wheel-install directory; 48 threads/rank; GEMM, LAPACK, autograd, shared NumPy conversion, MPI all-reduce and all-gather of hostnames PASS |
| Async MPI | Seven cases below on both nodes; pending operation observed before and after independent computation; correct results and exit 0 |
| GEMM benchmark | Complete FP32 and FP64 records, correctness PASS, application exit 0 for each 1/2/4-rank layout |

Fujitsu MPI writes rank outputs under `output.JOB_ID/0/1/stdout.1.RANK` in
the working directory. Scheduler `.out` and `.err` files may be in the job-script
submission directory and may contain little output. Check the recorded
application exit file as well as all expected rank records. PJM `EXT` means
finished, not necessarily successful; previous time-limited experiments even
left wrapper exit 0 with incomplete numerical output. Completeness matters.

`record-validation.py` deliberately rejects a missing rank, missing PASS marker,
nonzero exit, wrong rank count, incomplete dtype set or a purported two-node
test reporting only one host. Its JSON gate prevents historical result reuse.

## Asynchronous behavior tested

The installed wheel uses `init_process_group("mpi")`, not this repository's
separate `fjmpi` extension. `python-async.py` exercises:

1. `all_reduce(async_op=True)` with one peer delayed **three seconds**. The early
   rank must return promptly, observe unfinished work, do an independent GEMM,
   still observe unfinished work, then `wait()` and verify the reduced tensor.
2. Three queued all-reduces, separate retained buffers/work handles, and
   `get_future().wait()`, `done()` and correct future values.
3. Async broadcast.
4. Async all-gather.
5. Async barrier.
6. `irecv` before a deliberately delayed sender, completion polling with
   `is_completed()`, `wait()` and exact payload verification; matching `isend`.
7. Bidirectional preposted `irecv` plus `isend`, independent compute, both
   requests waited on and receive buffers checked.

Do not modify a send/collective buffer or inspect an unfinished receive buffer.
Keep buffers and work handles alive until completion. The test follows that
ownership rule. Tiny messages can complete immediately, so an early completed
flag alone would not disprove asynchronous behavior; delayed peers make the
pending states observable. The artificial-delay timings are not network-latency
or training-speedup measurements.

At this PyTorch pin, collective work runs in a **single ProcessGroupMPI worker
thread calling blocking `MPI_Allreduce`, `MPI_Bcast`, etc.** The Python API
returns asynchronously, but this is not `MPI_Iallreduce` and does not establish
multiple simultaneous MPI collectives. Native point-to-point work uses
`MPI_Isend`/`MPI_Irecv`, `MPI_Test` and `MPI_Wait`. The initialized thread level
was MPI_THREAD_SERIALIZED (2); MPI calls are protected by a process-global
mutex. P2P futures and timeout guarantees were not tested; the P2P implementation
does not have the same WorkMPI future path as collectives. Its wait timeout
argument is ignored in the inspected implementation. Keep scheduler limits.
The built-in MPI backend also leaves some APIs unsupported, including
`allreduce_coalesced` at this pin. Do not assume the capability set or the
custom reduce-scatter strategy of this repository's separate `fjmpi` extension
applies to the new wheel.

## Performance measurement scope

Each 4096x4096 benchmark rank owns independent matrices. It warms up one GEMM,
then measures the mean of three GEMMs. The reported time is the maximum of
those per-rank means, and aggregate TFLOP/s is
`ranks * 2 * 4096**3 / max_rank_seconds / 1e12`. Every rank checks the output.
Barriers and timing reductions use MPI, but communication is outside the timed
GEMM. This is a node GEMM test, **not** distributed matrix multiplication,
multi-node scaling, a DDP training test or an overlap-speedup benchmark.

Submit **separate** one-node jobs for 1x48, 2x24 and 4x12 ranks/threads. Do not
request four ranks per node and then assume an `mpiexec -n 1` subrun has the same
binding as a true one-rank allocation. The Python runs were a first small sample;
different nodes, initialization/first-touch, binding, runtime environment and
mean-versus-median statistics limit direct comparison with earlier native probes.
