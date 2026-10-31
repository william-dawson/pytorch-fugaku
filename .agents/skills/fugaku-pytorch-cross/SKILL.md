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

## Supplied material

`assets/recipe/` preserves the tested build scripts, CMake files, five patches,
Python wheel assembler and runtime tests. These files contain historical paths;
**stage them with `scripts/prepare.py` rather than executing them in this skill directory**.
The preparer substitutes explicit site paths, supplies guarded launchers and
isolates the Python 3.11 host generator environment. Those staging adaptations
have local validation; they are not a claim of a second complete Fugaku rebuild.
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
