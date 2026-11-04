---
name: fugaku-pytorch-cross
description: Cross-compile and validate PyTorch for Fugaku A64FX using LLVM, Fujitsu MPI, threaded BLAS/LAPACK and Python bindings. Use for reproducing this source-build route or diagnosing its build/runtime failures.
---

# LLVM cross-compilation of PyTorch for Fugaku

Use this for the repository's LLVM source-build route. The existing container,
`torch_fugaku` extension and its `fjmpi` backend are separate implementations.
This route builds PyTorch's own `torch.distributed` backend named **`mpi`**.

The validated baseline is PyTorch **2.11.0**, LLVM **22.1.0**, Fujitsu TCS/DSP
**1.2.43**, A64FX/SVE512, LP64 threaded `fjlapackexsve`, Fujitsu OpenMP, and
CPython **3.11.5 AArch64**. Sources and wheel checksums are pinned in
[assets/inputs.lock.json](assets/inputs.lock.json).

## Route the work

- For compiler, ABI and runtime decisions, read [references/toolchain.md](references/toolchain.md).
- For a fresh build, follow [references/reproduce.md](references/reproduce.md). It explains the
  prepared workspace and the separate LibTorch and Python phases.
- For failure diagnosis, use [references/troubleshooting.md](references/troubleshooting.md).
- For compute-node tests, asynchronous MPI semantics and benchmarks, read
  [references/validation.md](references/validation.md).
- For what has actually passed, including limits, use [references/results.md](references/results.md)
  and its small machine-readable evidence files. Historical PASS records are not fresh validation.

Read [references/large-pages.md](references/large-pages.md) for the required
system allocator, linked Python runner and library-path isolation.

**For training performance, read [references/performance.md](references/performance.md) first.**
The large-page build as built trains 10x slower than the container. Three run-time settings, no
rebuild, make it match the container at 125M and beat it by 15% at 1B:
`#PJM --llio cn-cache-size=384Mi`, the `fjmpi_port` backend (container's `c10d_mpi.cpp` built
against the wheel) instead of `mpi`, and mimalloc preloaded into the Python process at 1B scale.
The page also lists every dead end and the measurement pitfalls. Harness and backends are in
`assets/h2h/`, `assets/backends/`, `assets/alloc/`. Communication overlap (Fujitsu persistent
collectives, the experimental `fjmpi_async`) is in [references/overlap.md](references/overlap.md).
Benchmarks against the containers: benchkit PR 343 and the legacy Fugaku-LLM Megatron SIF
(how to run it, which sizes are comparable), in [references/comparisons.md](references/comparisons.md).
**Run the LLVM build at 4 ranks x 12 threads per node only** (it is ~40x slow with wider ranks).
For models with dropout, load `assets/fastdrop` (stock CPU `bernoulli_` was 71% of a GPT step).
The same kernel is now in torch_fugaku, so the new SIF gets it too (then LLVM leads by 2% at
GPT-125M, 15% at 1.3B). With both, it beats the legacy SIF (1.5x at GPT-125M, 2.5x at 1.3B) and the new SIF (5-25%;
benchkit Llama-1B fits 4x12 only at 12 and 16 nodes, 239 and 225 tokens/s/node).
From 64 ranks, the fjmpi backends pick their reduce-scatter (`TORCH_FUGAKU_RS_ALGO=auto`): ring at
4 ranks/node (-8% to -13% step time at 16 nodes), one all-to-all at 2 ranks/node (-1.4%). This needs
torch_fugaku after this change: the b5879f2 image predates it. See performance.md, Fix 5.
**To scale, use the container** ([references/scaling.md](references/scaling.md)): 4x12 with
llama_train's `--loss-chunk 256` gives 184 -> 164 tokens/s/node from 16 to 128 nodes, with
communication at 12-16%. 2x24 and 1x48 lose more with scale (reduce-scatter grows ~4x) and compute
less per node, but have the memory headroom. `TORCH_FUGAKU_COMM_STATS=2` breaks a step down.

## Operational invariants

1. Cross-compile on an **x86 login node**. Run AArch64 artifacts and the target
   Python only in compute-node jobs. Build tools and code generators must be x86 executables.
2. Build with at most **8 concurrent compiler processes** for this recipe.
   Use a fresh workspace in the project data area; the home quota is unsuitable.
3. Keep `USE_DISTRIBUTED`, `USE_MPI`, `USE_C10D_MPI` and `USE_OPENMP` enabled.
   Configuration may silently turn MPI or Python off: run the provided guards.
4. Use Fujitsu's active OpenMP runtime with threaded Fujitsu BLAS. Do not let
   Clang add LLVM `libomp` alongside it. Verify ELF dependencies and actual runtime behavior.
5. Both launchers acquire the **same filesystem lock**, inherited by the build,
   and record host plus PID. A PID on one login host says nothing about another
   host. Never start a second build against an active build directory or signal
   processes just because their PIDs match an old record.
6. A successful build is followed by real compute-node tests. PJM `EXT` is not
   an application success code. Require the recorded exit code and complete rank outputs.
7. Performance comparisons run the two stacks concurrently or in one allocation, with all
   instrumentation that walks page tables or runs inside a rank's CPU set switched off
   (see performance.md, "Measurement pitfalls"). Spread between allocations reaches ~10%.

## Supplied material

`assets/recipe/` preserves the tested build scripts, CMake files, eight patches,
Python wheel assembler and runtime tests. These files contain historical paths;
**stage them with `scripts/prepare.py` rather than executing them in this skill directory**.
The preparer substitutes explicit site paths, supplies guarded launchers and
isolates the Python 3.11 host generator environment. The November 2026 fresh rebuild passed core and Python compute-node validation.
See [references/large-pages.md](references/large-pages.md) for the exact evidence
and the distinction between tested commands and updated staging wrappers.
Run `scripts/validate-stage.py PREPARED_WORKSPACE` before transfer/build for
checksum, source-patch round-trip, syntax and target NumPy architecture checks.
`scripts/test-record-validation.py` exercises incomplete-output rejection.

`scripts/fetch-inputs.py` downloads public pinned sources/wheels to a user-selected
cache. `scripts/prepare.py` verifies that cache and exports committed source
content to a new directory without copying build artifacts or old validation.
`scripts/record-validation.py` records only complete outputs from new jobs.

Follow the current session's authorization and HPC tool rules. This skill does
not grant permission to submit point-consuming jobs, send signals, change SSH
authentication, or push commits. Reuse authorization already given in the session.

Do not treat an inconclusive check as permission to restart. If a build host
cannot be reached, monitor shared files and report that limitation. Preserve
failure logs before a retry; stop and diagnose repeated failures rather than
launching overlapping copies.
