# Collective microbenchmark and the reduce-scatter switch

Used for "Fix 5" in [references/performance.md](../../references/performance.md).

## Files

- `rs_bench.cpp`: reduce-scatter and allgather built from different MPI primitives, one variant
  per process. Build on the login node:
  `mpiFCCpx -Nclang -std=c++17 -Kfast -Kopenmp rs_bench.cpp -o rs_bench`.
  - Reduce-scatter variants: `a2a_staged` (fjmpi's old path), `a2a`, `rsb`
    (`MPI_Reduce_scatter_block`), `ring`, `halving`, `allreduce`.
  - Allgather variants: `ag_staged`, `ag`, `ag_ring`, `ag_persist` (`MPIX_Allgather_init`).
  - Each variant checks its result exactly (the inputs sum exactly in float32). It prints one JSON
    line per size: time per call (median of reps, max over ranks) and node `MemAvailable` before
    and after.
- `run.sh`: one job. Every variant in `VARIANTS` gets its own `mpiexec` with 4M and 50M floats per
  call. Set `RPN` (ranks per node) and `#PJM --mpi max-proc-per-node` to match.
- `build_container.sh`: builds `torch_fugaku/csrc/c10d_mpi.cpp` inside the pytorch-fugaku image
  into `$WORK/h2h-rsalgo/container/torch_fugaku`: the image's package, with a new `_c10d.so`.
  This is for testing without rebuilding the image. The LLVM build of the same file is
  [../backends/fjmpi_port](../backends/fjmpi_port) with `c10d_mpi.cpp` replaced.
- `test.sh`: a 2-node job. It builds the container backend, then runs `tests/test_backend_mpi.py`
  under each `TORCH_FUGAKU_RS_ALGO`.
