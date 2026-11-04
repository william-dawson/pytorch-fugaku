# Training performance: from 10x slower to faster than the container (November 2026)

The large-page build in [large-pages.md](large-pages.md) trained 10x slower than the
pytorch-fugaku container (container 2.4 s/step, LLVM 10-45 s/step, 151M Llama, ZeRO-3,
2 nodes x 4 ranks x 12 threads). Three independent causes were found and fixed **without
rebuilding PyTorch or using the container**. Each was hidden by the one before it. With all three,
the LLVM build matches the container at 125M and beats it by 15% at 1B.

Everything here was measured with the harness in `assets/h2h/` (see its README). Job IDs refer to
the `ra000009` runs; raw per-rank timings stayed on Fugaku.

## Results

FP32, DeepSpeed 0.19.8 ZeRO-3, sequence 128, micro-batch 1, activation checkpointing, `--low-mem`,
4 ranks x 12 threads per node, `freq=2000,eco_state=0`. Steady step = median over steps after
warmup of the slowest rank. Losses match the container to 4 decimals on every step in every row.

| Model, nodes | Container SIF | LLVM build, all fixes | Jobs |
|---|---:|---:|---|
| llama-125m (151M), 2 | 2.40 s | 2.34 s | 53367016 / 53411647 |
| llama-125m, 8 | 2.80 s | 2.70 s | 53463780 (same allocation) |
| 1b (1.24B), 4 | 11.16 s | **9.52 s** | 53477800 / 53477818 (concurrent) |

The 1B row uses mimalloc (fix 3); at 125M the build's own libmpg allocator already gives zero
page faults and mimalloc was not needed.

Run-to-run spread between allocations is up to ~10% (the container measured 10.27, 11.16 and
11.45 s/step at 1B in different jobs). Compare cases that ran concurrently or in one allocation.

## The recipe

```sh
#PJM -L "rscgrp=small"            # f-pt was used for fast turnaround
#PJM -L "node=4"
#PJM -L "freq=2000,eco_state=0"
#PJM --mpi "max-proc-per-node=4"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM --llio cn-cache-size=384Mi    # fix 1
```

Per rank (from `assets/h2h/rank.sh`, case `llvm-port+mi`):

```sh
export OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PROC_BIND=false OMP_WAIT_POLICY=ACTIVE
export RANK=$PMIX_RANK LOCAL_RANK=$((PMIX_RANK % 4)) LOCAL_SIZE=4 USE_LIBUV=0
export TORCH_FUGAKU_STAGE_MB=8        # fjmpi_port staging buffers
# WORLD_SIZE, MASTER_ADDR, MASTER_PORT from fugaku/pf-setup's pf.env (env:// rendezvous)
env LD_PRELOAD=/path/to/libmimalloc.so.2.2 \
  rebuild-largepage/bin/python3.11-fugaku train.py ...   # fix 3; python only, not mpiexec
```

and in Python, before DeepSpeed (fix 2):

```python
import fjmpi_port; fjmpi_port.register_backend()     # assets/backends/fjmpi_port
deepspeed.init_distributed(dist_backend="fjmpi_port")
```

## Fix 1: LLIO's compute-node cache was too small for PyTorch's code

**Symptom.** About 900 major page faults per backward pass, 10-45 s/step, never settling.

**Diagnosis.** The existing `perf record -e major-faults -d` capture (job 53270101) holds the
faulting *data address* of every sample. It had never been printed. `assets/fault-addr/` prints
it with `perf script -F tid,time,addr,ip,sym,dso` and maps each address onto the rank's
`/proc/<pid>/maps` (job 53393819). Result: essentially all faults are on file-backed **executable
code**, about 71% in `libtorch_cpu.so`, 15% in `libtorch_python.so`, 5% in the interpreter and 4%
in `libc10`. The same pages fault over and over: rank 0's backward passes took 4607 faults on only
204 distinct 64 KiB pages; one Sleef `exp` page faulted 649 times. Memory was nowhere near full.

**Cause.** `/vol000x` (and `$PJM_LOCALTMP`) are LLIO file systems. LLIO keeps file data in its own
per-node cache, sized by `pjsub --llio cn-cache-size` (visible in a job as
`PJM_LLIO_CN_CACHE_SIZE`). The default for this group is **128 MiB** (`pjacl --rg f-pt`; allowed
4Mi-32768Mi). `libtorch_cpu.so` alone has 127 MB of code. The read-only segments of all loaded
libraries total 294 MB per process (measured by `mlock`ing them, job 53390380). The working
inference is that hot code evicts itself from the LLIO cache and the kernel's mapped pages go with
it. It was not traced inside LLIO, but raising the cache removed the faults.

**Fix.** `#PJM --llio cn-cache-size=384Mi`: just above what is needed, not a blanket 4 GiB.
Faults after step 0 drop to 0 (job 53402434). Size it from the sum of the read-only segments of
the loaded `.so` files plus headroom; larger workloads that read more files may need more.

The container never hit this: its libraries live inside the SIF's SquashFS image.

## Fix 2: torch's ProcessGroupMPI stalls; use the container's backend

**Symptom.** With faults gone, backward passes stalled 6-290 s (job 53402434).

**Diagnosis.** `perf record -e cpu-clock -g` on ranks 0 and 4 (job 53405897): 92% of CPU in
`libfjomp`'s fork-barrier spin (idle OpenMP workers), about one thread polling `libmpi`/
`libtofucom` (ProcessGroupMPI's worker thread), under 0.5% compute: every rank waiting on
collectives. The faults had hidden this: threads blocked in page faults left cores free.

`OMP_WAIT_POLICY=PASSIVE` made it **worse** (uniform 104.7 s/step, job 53404112). Keep ACTIVE.

**Fix.** `fjmpi_port`: this repository's `torch_fugaku/csrc/c10d_mpi.cpp` (byte-identical),
compiled against the LLVM wheel as a separate backend name. Build glue is in
`assets/backends/fjmpi_port/`. It needs `env://` rendezvous variables (see the recipe);
without them: `environment variable RANK expected`. Result: 2.34 s/step vs the container's
2.40 s/step (job 53411647).

The two designs differ in how they call MPI:
- **ProcessGroupMPI** queues every collective onto one background thread that calls blocking MPI. Its "async" means another thread blocks for you.
- **fjmpi** calls Fujitsu MPI's blocking collectives on the calling thread, through two reused staging buffers.

A microbenchmark (`assets/overlap/`, the `thread` variant) reproduces the first pattern without
torch. It gave only 12-40% overlap, and at 8 MB per rank it was slower than doing it serially:
the MPI thread competes with 12 spinning OpenMP threads for 12 cores.
The exact mechanism of the multi-second stalls was not isolated.

## Fix 3 (1B scale): the allocator. Preload mimalloc

**Symptom.** With libmpg's large pages the LLVM build runs out of memory at 1B on 4 nodes
(SIGKILL, exit 137) where the container fits. With large pages off
(`XOS_MMM_L_HPAGE_TYPE=none`) it fits but runs 12.6 vs 10.3 s/step.

**Diagnosis.**
- **Operator profile** (torch.profiler on two steady steps, `H2H_PROFILE_STEPS`, job 53467979, `compare_profiles.py`):
  - `aten::mm` was identical (3.70 vs 3.73 s).
  - The extra time was in allgather (+1.25 s), `AccumulateGrad` (+0.96 s, 4.5x), DeepSpeed's `PostBackwardFunctionModuleBackward` (+0.53 s), reduce-scatter (+0.53 s) and `cat` (+0.20 s). All of these first-touch freshly allocated buffers.
- **Phase counters** (`getrusage` per forward/backward/step, always on in `run.py`):

| 1B, per steady phase (mean over ranks) | Container | LLVM, large pages off |
|---|---:|---:|
| Forward minor faults / system CPU | 4.8k / 0.17 s | 106k / 1.33 s |
| Backward minor faults / system CPU | 8.2k / 0.26 s | 233k / 3.34 s |

At 125M with large pages on the LLVM build takes 0 faults, so the allocator, not the model, decides this.

**What did not work** (all 1B, 4 nodes):

| Setting | Result | Job |
|---|---|---|
| libmpg `XOS_MMM_L_ARENA_FREE=1` or `2`, large pages off | no change (232k faults) | 53469399 |
| same, large pages on | out of memory at step 0 | 53469399 |
| plain miniconda `python3.11` (no libmpg; host glibc 2.28) | out of memory in step-1 backward | 53471811 |
| libmpg + `MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=64G` | out of memory | 53471811 |
| plain python + `MALLOC_ARENA_MAX=1` or `2` | out of memory | 53473242 / 53473260 |
| mimalloc with `MIMALLOC_PURGE_DELAY=-1` (never return memory) | bus error in MPI at step 0 (out of memory) | 53474731 / 53474746 |

Per-phase peak RSS (`H2H_PEAK=1`, job 53473997) showed the LLVM ranks are **not** larger than the
container's (6.6 vs 6.8 GB). The difference is how freed memory is treated:
- **Container's glibc:** keeps freed memory and reuses it. RSS stays flat at 6.8 GB, with few faults.
- **libmpg (large pages off):** returns freed memory to the OS. RSS swings about 2 GB inside every phase, and the memory is faulted back in each time.
- **Host glibc and libmpg large pages:** keep memory but grow past the node.
Huge pages were ruled out: every process on both stacks holds the same 48 MB of them.

**Fix.** Preload mimalloc 2.2.6, built from PyTorch's own pinned `third_party/mimalloc`
(`assets/alloc/build-mimalloc.sh`), into the python process only, at its default purge delay:
- A preloaded `malloc` takes precedence over libmpg's, which is only a dependency of the runner. It also covers torch's `posix_memalign`, which libmpg does not override.
- Results, concurrent with the container (jobs 53477800 / 53477818 / 53477832):

| 1B, 4 nodes | Step | Forward | Backward | Backward faults |
|---|---:|---:|---:|---:|
| Container | 11.16 s | 2.97 s | 7.86 s | 8k |
| **LLVM + mimalloc** | **9.52 s** | 2.55 s | 6.62 s | 10k |
| LLVM + mimalloc, large pages off | 9.80 s | 2.70 s | 6.76 s | 11k |

This complements `system-allocator.patch`, which keeps mimalloc out of `libc10` itself (PyTorch
2.11 forces it on Linux/AArch64). The preload replaces the process allocator at run time and
needs no rebuild.

## Fix 4 (models with dropout): fast CPU `bernoulli_`

**Symptom.** GPT-style models train 2-4× slower than expected: GPT-125M ran at 32.1 s/step, but
8.07 s/step with dropout off (jobs 53487645 / 53488227).

**Diagnosis.** In a torch.profiler step, `aten::bernoulli_` (dropout's mask) took 22.8 s of 32.1 s
(71%). Stock PyTorch's CPU `bernoulli_` draws random numbers one element at a time. Fujitsu's
PyTorch 1.13 fork needs 0.69 s for the same 73 calls. See [comparisons.md](comparisons.md).

**Fix.** `assets/fastdrop/libfastdrop.so` replaces `aten::bernoulli_.float` on CPU at load time
(`torch.ops.load_library`; in the harness, `H2H_FASTDROP=<path>`).
- **Generator.** Element i of a call gets a keyed integer hash of i, so the loop has no carried
  state and vectorizes and parallelizes.
- **Seeding.** The key is one 64-bit draw from torch's generator per call, so `torch.manual_seed`
  still makes runs reproducible.
- **Types.** float32 directly; other float types through a float32 temporary.
- **Unit test** (`test_fastdrop.py`, job 53493367, 12 threads, 12.6M elements):
  `bernoulli_` 210.6 → 1.85 ms (114×), `F.dropout` 216.4 → 7.6 ms (28×). Means at p = 0-1 are
  within 6σ, the stream is reproducible under `manual_seed`, consecutive calls differ in 50% of
  elements, lag-1 correlation is 0.0002, and float64, bfloat16 and non-contiguous tensors pass.
- **Training:** GPT-125M 32.1 → 9.54 s/step (53493468); GPT-1.3B 4 × 12 ZeRO-3 40.2 → 23.4 s/step
  (53493472).

The random stream differs from stock PyTorch's, so losses are not bit-identical to the container,
but they fall normally. Llama has no dropout and is unaffected.

## Fix 5 (64+ ranks): reduce-scatter algorithm (`TORCH_FUGAKU_RS_ALGO`)

**Symptom.** At scale, reduce-scatter dominated the exposed communication: 6.5 s of a 32 s step for
benchkit Llama-1B at 64 nodes, 2 × 24 (job 53511140).

**Diagnosis.** fjmpi reduce-scatter is staged: chunks of the 8 MB staging buffer go through
`MPI_Alltoall`, then a local sum. Each peer's slice is 8 MB / (4 B × ranks), so at 128 ranks a
200 MB call becomes 24 latency-bound all-to-alls. `assets/collectives/rs_bench` measured each
alternative in its own `mpiexec`. Times are seconds per 50M-float call; every result was checked
exactly:

| variant | 2×24, 32 ranks | 4×12, 64 ranks | 4×12, 128 ranks | 2×24, 128 ranks | 2×24, 256 ranks |
|---|---:|---:|---:|---:|---:|
| `a2a_staged` (old fjmpi) | **0.16** | 0.39 | 0.68 | 3.41 | 4.77 |
| `a2a` (one `MPI_Alltoall`) | 0.17 | 0.20 | 0.33 | **0.38** | **0.48** |
| `ring` (`MPI_Sendrecv` to neighbours) | 2.52 | **0.095** | **0.135** | 15.3 | > 36 at 4M floats |
| `halving` (recursive) | 0.32 | 0.26 | 0.35 | 0.95 | timed out |
| `allreduce` + slice | 0.21 | 0.37 | 0.40 | 46.1 | timed out |
| `rsb` (`MPI_Reduce_scatter_block`) | 0.55 | 0.50 | 0.55 | 0.54 | 0.56 |

Jobs 53512513-53512527; data in `evidence/collectives-2026-11-04.json`.
- **At 4 ranks per node, ring wins** by 4-5× from 64 ranks up.
- **At 2 ranks per node, anything that reduces between exchanges collapses at scale.** That
  includes ring, halving and Fujitsu's own `MPI_Allreduce`. Only the all-to-alls and `rsb` hold up.
  Ring *allgather* is fine at 2 × 24 (below), so plain point-to-point is not the problem.
- **No variant grows MPI's memory**, so reduce-scatter is not why 4 × 12 runs out of memory at
  some node counts.
- **Allgather does not collapse.** Staged 0.053-0.066 s, one `MPI_Allgather` 0.031-0.045 s,
  persistent 0.049-0.071 s, ring 0.11-0.15 s, at 64-128 ranks on both layouts (jobs
  53576485-53576491). It is left staged.

**Fix.** `torch_fugaku/csrc/c10d_mpi.cpp` takes `TORCH_FUGAKU_RS_ALGO=auto|staged|a2a|ring`
(default `auto`). Both the image's `fjmpi` and fjmpi_port build this file.
- `auto` uses `staged` below 64 ranks, `ring` at 4 or more ranks per node (counted with
  `MPI_Comm_split_type`), and `a2a` otherwise.
- Ring runs through the two staging buffers, so Tofu never registers new memory.
- Rank 0 logs the choice: `fjmpi: reduce_scatter uses ring (64 ranks, 4 per node)`.
- `tests/test_backend_mpi.py` passes under all four settings (job 53575982).

**Training** (benchkit Llama-1B, seq 2048, ZeRO-3; each pair ran in one allocation; losses
identical within each pair):

| stack, layout | staged | new | tokens/s/node |
|---|---:|---:|---:|
| container, 4 × 12, 16 nodes | 48.05 s | ring 44.02 s (−8.4%) | 170.5 → 186.1 |
| LLVM (+mi), 4 × 12, 16 nodes | 57.05 s | ring 49.65 s (−13%) | 143.6 → 165.0 |
| container, 2 × 24, 64 nodes | 34.75 s | a2a 34.25 s (−1.4%) | 117.9 → 119.6 |

The 4 × 12 allocations (jobs 53576418 / 53576423) were slow overall. Earlier 16-node runs did about
36 s (LLVM) and 40 s (container) per step, so compare within the pairs only. At 2 × 24 the
microbenchmark's 9× did not carry over. Either registering torch's fresh buffers costs it back, or
most of the profiled reduce-scatter time was waiting on other ranks. This is not yet profiled.

## Layout: 4 ranks × 12 threads only

The LLVM build is about 40× slower with fewer than 4 ranks per node (2 × 24, 1 × 48): collectives
spin for about 1.4 s each. Six variations (allocator, staging, thread count, wait policy) did not
fix it; see [comparisons.md](comparisons.md). Use one rank per CMG. Where 4 × 12 runs out of
memory (benchkit Llama-1B at 4 and 32 nodes), the LLVM build has no working layout yet.

## Dead ends (do not repeat)

| Tried | Outcome | Job |
|---|---|---|
| `mlockall(MCL_CURRENT\|MCL_FUTURE)` | every rank SIGKILLed | 53389437 |
| `mlock` of the 294 MB of `.so` read-only segments | locked (VmLck 287 MB) yet faults unchanged | 53390380 |
| copy the tree to `/dev/shm` | `cp -a` of 26k small files took 975 s; then `/dev/shm` is **noexec** on compute nodes | 53394472 / 53397266 |
| run from `$PJM_LOCALTMP` (`--llio localtmp-size`) | also LLIO; faults unchanged | 53398035 |
| run from `/2ndfs` (direct FEFS) | fast step 0, then 53-364 s stalls (fix 2's problem) | 53399014 |
| `cn-cache-size=4Gi` | same stalls; also far more than needed | 53400881 |
| `OMP_WAIT_POLICY=PASSIVE` | 104.7 s/step | 53404112 |

The earlier "10x RSS difference" between stacks was accounting: libmpg's heap is in hugetlb pages,
which VmRSS does not count. See large-pages.md.

## Measurement pitfalls found the hard way

- **Never run a sampler inside a rank's CPU set.** A 2 s `ps`/`awk` loop started from one rank per node preempted that rank's spinning OpenMP threads. Every collective then waited for it: step times jittered 10.5-20 s against a steady 9.5 s (job 53475550 vs 53477818). The giveaway is involuntary context switches concentrated on ranks 0, 4, 8 and 12. The harness keeps this opt-in (`H2H_MEMSAMPLE=1`).
- **Resetting VmHWM costs time.** Writing `5` to `/proc/self/clear_refs` resets the high-water mark but walks the page tables. Doing it every phase moved the container from 11 to 15 s/step (job 53473983). It is opt-in (`H2H_PEAK=1`).
- **A rank killed for memory never writes its end-of-run JSON.** `run.py` also prints one `{"event": "phase", ...}` line per phase to the rank log as it happens.
- **Killed runs leave little trace.** A bus error with "non-existent physical address" inside MPI, or exit 137, means the node ran out of memory. PJM reported no per-node memory limit (`pjstat -v`: unlimited).

## Fujitsu facts this relied on (paraphrased; see the cited manuals)

- LLIO compute-node cache: `pjsub --llio cn-cache-size`, per-group limits via `pjacl`.
- `/dev/shm` on compute nodes does not allow execution; `$PJM_LOCALTMP` is LLIO-backed.
- libmpg (`/opt/FJSVxos/mmm/lib64`) overrides `malloc`/`free`/`realloc` (not `posix_memalign`);
  `XOS_MMM_L_HPAGE_TYPE=none` turns its large pages off. Its full option list is in Fujitsu's
  "Job Operation Software End-user's Guide for HPC Extensions", which is not on the system.
- Fujitsu MPI's non-blocking progress, assistant cores and persistent collectives:
  see [overlap.md](overlap.md).

## Open items

- 8+ node 1B, longer runs, and mimalloc at 125M (not needed there, not yet shown harmless).
- Why wide ranks (2 × 24, 1 × 48) make every collective cost about 1.4 s.
- Why reductions collapse at 2 ranks per node (Fix 5), and why the a2a gain at 2 × 24 is small in training.
- Container scaling across layouts, memory and communication: [scaling.md](scaling.md).
- The mechanism of ProcessGroupMPI's stalls (fix 2 avoids it rather than explaining it).
- Communication is still exposed: 30% of a 125M step and about half of a 1B step is spent blocked
  in allgather and reduce-scatter. See [overlap.md](overlap.md).
