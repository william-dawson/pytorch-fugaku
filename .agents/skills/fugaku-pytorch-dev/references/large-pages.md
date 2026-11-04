# Fresh allocator build, November 2026

## What changed and why

The compiler remains LLVM 22.1.0, targeting A64FX/SVE512, with effective `-O3`,
Fujitsu MPI, LP64 threaded `fjlapackexsve` and Fujitsu OpenMP. These choices were
already present. No compiler upgrade, global fast-math, software-pipeliner flag,
communication adapter, or container kernel port was part of this rebuild.

PyTorch 2.11's CMake unconditionally enables mimalloc on Linux/AArch64 even
when the cache says USE_MIMALLOC=OFF. `system-allocator.patch` adds an explicit
FUGAKU_USE_SYSTEM_ALLOCATOR opt-in; both build phases set it ON and mimalloc OFF.
The compile-command guard rejects any remaining `-DUSE_MIMALLOC`. A cache value
alone is not evidence of the allocator actually compiled into libc10.

The toolchain retains `libmpg` before other dependencies using scoped
`--no-as-needed`, with a compute-node RPATH. The cross library comes from
`/opt/FJSVxos/devkit/aarch64/rfs/opt/FJSVxos/mmm/lib64`; runtime uses
`/opt/FJSVxos/mmm/lib64`. No LD_PRELOAD is needed.

`python-runner.c` embeds the existing target CPython 3.11 via PyConfig and its
configured installation prefix. `preflight-build.sh` produces a non-PIE
`bin/python3.11-fugaku`, linking libmpg first, the complete Fujitsu runtime set,
and libpython. It applies `bss-2mb.lds` to the executable. Do not apply this
static-data linker script indiscriminately to Torch shared libraries. Linking
only libfjomp was insufficient: the initial preflight failed on
`__jwe_prsc_get_cpuid`; the supplied full runtime set resolved it.

Large pages cover eligible allocations/static data, not executable `.text`.
They therefore do not directly cure code-page faults. Default policy observed:
`demand:demand:prepage`. Inspect actual tensor pointers in smaps; neither a loaded
libmpg nor environment settings alone prove allocation coverage. Hugetlb usage
is recorded separately from ordinary RSS: the new training report's 0.23 GiB
RSS is NOT its total memory footprint. Account for Private/Shared_Hugetlb and
avoid double-counting shared mappings when comparing node memory.

Two supplemental CMake patches limit missing-prototype warning exemptions to
generated registration files, including RegisterMeta. Fresh builds exposed
paths the earlier functionalization-only match missed. They do not disable
warnings globally or change numerical flags. Apply patches in build.sh order.
Before a resume, remove overlapping supplemental patches in reverse order;
otherwise the base patch's context check fails. Preserve failure logs and
never overlap builds. All launchers share a filesystem lock; MAX_JOBS <= 8.

## Compute-node evidence

- Allocator preflight 53279299: linked runner, 256 MiB posix_memalign allocation,
  2048 kB kernel/MMU pages, nonzero Private_Hugetlb.
- Core 53363613: two distinct nodes; GEMM, LAPACK, autograd, native MPI and
  OpenMP PASS, build/application exit 0.
- Python 53366129: both ranks PASS actual 256 MiB Torch tensor allocation on
  2048 kB pages; containing mapping Private_Hugetlb 264192 kB. Mappings contain
  libmpg, threaded Fujitsu BLAS and Fujitsu OpenMP, without libomp/libgomp/libiomp.
  GEMM, LAPACK, autograd, NumPy and MPI PASS; all seven async cases PASS.
  The allocation/async suites used 12 threads; the inherited smoke script used
  48. The updated generic smoke helper now honors OMP_NUM_THREADS.
- The delayed-peer all-reduce returned on rank 0 in 0.241 ms with the peer
  delayed 3 s, and independent GEMM ran while the request was pending. This
  proves nonblocking API behavior, not optimized training communication.

The successful core command used:

```sh
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE
LD_LIBRARY_PATH=/opt/FJSVxtclanga/tcsds-1.2.43/lib64 \
  mpiexec -n 2 /usr/bin/env -u LD_LIBRARY_PATH ./build-smoke/torch-mpi-smoke
```

The Python job used the same launcher/environment separation and the new runner
for `validate-new-runtime.py`, then `test-fresh-python.py python-smoke.py` and
`test-fresh-python.py python-async.py`. Clearing the library path for mpiexec
itself fails before application startup (`libfjstring_internal.so` missing).
The generated wrappers encode this separation. No LLVM library is preloaded
into the application and no user Python installation is overwritten.

## First training comparison, not a speedup claim

**Resolved:** the slowdown below came from LLIO's 128 MiB compute-node cache, torch's
ProcessGroupMPI, and (at 1B) the allocator. With fixes the build matches or beats the container.
See [performance.md](performance.md). The tuning sequence at the end of this page was carried out
there. Step 2 (fault addresses) found the cause.

Same 151,020,288-parameter Llama, FP32, ZeRO-3, sequence 128, microbatch 1,
checkpointing, two nodes, four ranks per node and 12 threads each. Nine steps,
three warmups. Input hashes match across the fresh/container pair; initial
parameter hashes differ across stacks, as in the previous comparison.

| Mean over ranks/measured steps | Old cross-build 53264831 | Fresh 53366742 | Container 53367016 |
|---|---:|---:|---:|
| Global tokens/s (slowest-rank window) | 41.36 | 42.95 | 439.86 |
| Forward seconds | 1.642 | 3.521 | 0.726 |
| Backward seconds | 22.928 | 20.139 | 1.472 |
| Optimizer seconds | 0.156 | 0.148 | 0.125 |
| Major faults/backward | 750.98 | 874.38 | 0 |
| Process CPU-seconds/backward | 261.02 | 232.93 | 16.18 |

All eight ranks passed in these first instrumented cases. The old reversed
instrumented run achieved only 17.31 tokens/s; its plain runs ranged 39–91.
Thus 42.95 is within prior variation. Plain/reversed fresh cases were still
under review when this record was written. All 146 inspected rank-zero fresh
parameter shards use 2 MiB pages, unlike the old 64 KiB mappings.

Prior address-level profiling found 8999 selected major-fault accesses in
file-backed executable/read-only mappings, mostly libtorch_cpu/libtorch_python,
with repeated addresses across steps. Current phase counters show faults still
occur but do not independently identify their addresses or establish causality.
LLIO pre-staging alone previously showed no benefit with files already cached.

## Measurement-led tuning sequence

1. Complete plain/reversed comparisons and verify all ranks, inputs, loss,
   affinity and huge-page coverage. Preserve raw samples, including outliers.
2. Recollect fault addresses on the new build. Correlate long backward intervals
   with code mappings, main/worker CPU and wait state before choosing a fix.
3. If repeated code faults remain, compare identical binaries on shared storage
   versus a supported node-local placement, with identical preload policy and
   memory accounting. Check actual placement; LLIO caching is not process RAM
   residency. Prefaulting is a separate diagnostic, not a presumed solution.
4. If faults disappear but CPU time stays high, compare the exact hot Torch
   operators and shapes against the container's registered implementations;
   port one measured bottleneck at a time. Good large-GEMM performance does not
   establish performance of bmm, layout conversions, reductions or optimizer ops.
5. Only if traces show exposed communication or spin overhead, test collective
   size/algorithm changes or OpenMP wait policy independently. Keep 4x12 and
   rank/CMG placement fixed initially. Defer FSDP/framework changes until evidence
   identifies a framework bottleneck, so the comparison stays interpretable.
