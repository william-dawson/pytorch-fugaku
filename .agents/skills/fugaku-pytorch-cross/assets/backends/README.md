# c10d backends for the LLVM build

Both register a separate `torch.distributed` backend name next to the build's own `mpi`.
They need the `env://` variables `RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`; see
[assets/h2h/rank.sh](../h2h/rank.sh). Both load Fujitsu's `libmpi.so` from tcsds 1.2.43 at import
time. No Fujitsu file is copied here.

## fjmpi_port

This is the container's backend compiled against the LLVM wheel. Its source is this repository's
[`torch_fugaku/csrc/c10d_mpi.cpp`](../../../../../torch_fugaku/csrc/c10d_mpi.cpp), byte-identical
to what was built.

```sh
cp torch_fugaku/csrc/c10d_mpi.cpp .agents/skills/fugaku-pytorch-cross/assets/backends/fjmpi_port/
# copy the directory to $WORK/h2h-profile/port (build.sh takes WORK as ../..), then on a login node:
bash -lc 'bash build.sh'      # login shell: build.sh calls `module load`
```

It produces `fjmpi_port/_c10d.cpython-311-aarch64-linux-gnu.so`. Put the directory on `sys.path`,
then:

```python
import fjmpi_port; fjmpi_port.register_backend()
dist.init_process_group("fjmpi_port")
```

Use `TORCH_FUGAKU_STAGE_MB=8`. This replaced torch's `ProcessGroupMPI`
(references/performance.md, fix 2).

## fjmpi_async

This is `c10d_mpi_async.cpp`: fjmpi_port plus queued, Tofu-offloaded allgathers, with
`stats()` and `trace()`. It is experimental and **not faster yet**; see
[references/overlap.md](../../references/overlap.md). Build it the same way: the directory goes
at `$WORK/h2h-async/backend`, or point `H2H_ASYNC_PATH` at it.

Environment variables:
- `TORCH_FUGAKU_ASYNC=0|1`
- `TORCH_FUGAKU_AG_STAGE_MB` (default 64)
- `TORCH_FUGAKU_TRACE=1`
- `TORCH_FUGAKU_STAGE_MB`

`test/` holds the 2-node correctness and overlap test (`run.sh` runs async then sync).
