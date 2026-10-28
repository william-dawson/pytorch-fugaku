# pytorch-fugaku

PyTorch for [Fugaku](https://www.r-ccs.riken.jp/en/fugaku/) (Fujitsu A64FX) as a
container image: the official PyTorch aarch64 wheel plus **torch_fugaku**, a
small package that adds what A64FX needs through public PyTorch interfaces:

- **CPU kernels** (`TORCH_LIBRARY_IMPL`): `bmm` threaded over the batch (the
  wheel multiplies one slice at a time without MKL; 9-15x on small slices) and
  SVE `gelu`, `gelu_backward`, `tanh` from Arm Optimized Routines (the wheel's
  run near scalar speed on A64FX; 3-11x). Loaded only on CPUs with SVE.
- **Gradients over Fujitsu MPI**: a DDP communication hook
  (`register_comm_hook`) that sums gradient buckets with `MPI_Allreduce` over
  Tofu instead of gloo over TCP (2x at 4 nodes, 2.1x at 12). The process group
  stays gloo; only the gradient transport changes.
- **Settings**: oneDNN off (its ARM Compute Library path is ~3x slower in
  backward on A64FX), `OMP_WAIT_POLICY=ACTIVE`.

The image is built and tested by GitHub Actions and contains nothing from
Fujitsu. On Fugaku, two small scripts bind Fujitsu MPI in at run time.

```
ghcr.io/william-dawson/pytorch-fugaku:latest     # linux/arm64
```

## Using it on Fugaku

On a login node (Singularity pulls arm64 images on the x86 login node):

```sh
singularity pull --arch arm64 pytorch-fugaku.sif docker://ghcr.io/william-dawson/pytorch-fugaku:latest
git clone https://github.com/william-dawson/pytorch-fugaku.git   # for fugaku/pf-setup, fugaku/pf-run
```

A job, one rank per CMG (4 per node, 12 threads each):

```sh
#PJM -L node=4
#PJM -L freq=2000,eco_state=0          # normal mode; see below
#PJM --mpi max-proc-per-node=4
export OMP_NUM_THREADS=12
pytorch-fugaku/fugaku/pf-setup                     # once per job
mpiexec -n 16 pytorch-fugaku/fugaku/pf-run pytorch-fugaku.sif python train.py
```

`train.py` needs nothing Fugaku-specific: `init_process_group("gloo")` and
`DistributedDataParallel` as usual. `pf-setup` provides `WORLD_SIZE`,
`MASTER_ADDR` (the first node's Tofu address), `MASTER_PORT` and
`GLOO_SOCKET_IFNAME`; `pf-run` adds `RANK` and `LOCAL_RANK` from Fujitsu
`mpiexec`. With `TORCH_FUGAKU=1` (the image default) torch_fugaku applies
itself when torch is imported, including the MPI hook on every DDP model when
the ranks were started by mpiexec.

Things to know:

- **Normal mode.** Fugaku jobs default to boost-eco (2.2 GHz, one FP pipeline
  per core in use). Normal mode (`-L freq=2000,eco_state=0`) makes well-shaped
  GEMMs ~1.4x faster; end to end it measured +9% (below).
- **DataLoader workers**: use `num_workers=0`. Fujitsu MPI's registered memory
  is not fork-safe; workers forked after `MPI_Init` crash.
- **Memory**: 32 GiB HBM2 per node, 8 GiB per CMG-sized rank.
- **`PF_BIND`** adds bind mounts (data, code), comma-separated.

## The Python interface

Everything the autoload does is available explicitly (set `TORCH_FUGAKU=0` to
turn the autoload off):

```python
import torch, torch_fugaku
torch_fugaku.enable()              # kernels and settings
model = torch_fugaku.ddp(model)    # DistributedDataParallel with gradients over MPI
torch_fugaku.status()              # {'kernels': ['parbmm', 'fastmath'], 'mpi_world': 16, 'sve': True}
```

Or keep your own DDP and register the hook: `model.register_comm_hook(None,
torch_fugaku.mpi_comm_hook)`. Environment switches for the autoload:
`TORCH_FUGAKU_KERNELS=0`, `TORCH_FUGAKU_ONEDNN=1`, `TORCH_FUGAKU_MPI=0|1|auto`
(auto: on when an MPI launcher started the process and `WORLD_SIZE > 1`).

## Measured on Fugaku (2026-10-28)

Kernels, one CMG (12 threads), 16 M float32 elements:

| op | stock wheel | torch_fugaku |
|---|---:|---:|
| `gelu` | 17.1 ms, 180 ULP | 5.3 ms, 89 ULP |
| `gelu_backward` | 29.7 ms, 380 ULP | 3.9 ms, 121 ULP |
| `tanh` | 43.5 ms, 1.0 ULP | 4.0 ms, 2.5 ULP |
| `bmm`, 4096 slices of 32x32x32 | 111 ms | 7.2 ms, bit-identical |

End to end, the [OmniLearned](https://github.com/ViniciusMikuni/OmniLearned) pretraining
step (PET v2 medium, FP32, batch 8 per rank, losses identical to the stock
wheel's), events/s:

| | 1 node | 4 nodes | 12 nodes |
|---|---:|---:|---:|
| stock wheel, gloo | 0.49 | - | - |
| torch_fugaku, boost-eco | 14.4 | 51.4 | 139.4 |
| torch_fugaku, normal mode | 16.0 | | |

(1 node: one epoch of 25 iterations; the 4- and 12-node rows are the BenchKit
two-epoch measurement. These were measured with the same kernels and hook
built outside this image; the image itself is being validated.)

## Fujitsu's linear algebra (optional)

The wheel uses NVPL for BLAS. Fujitsu's threaded BLAS (`libfjlapackexsve`)
beats it on some shapes, 48 threads, SGEMM GFLOP/s:

| shape | NVPL | Fujitsu |
|---|---:|---:|
| 3840 x 1536 x 512 | 2,858 | 1,939 |
| 4096^3 | 2,360 | 2,982 |
| 16384^3 | 2,461 | 3,008 |

and at 12 threads (one CMG) on most of OmniLearned's shapes. Whether it pays
depends on your shapes. Fujitsu's threaded BLAS needs the whole process on
Fujitsu's OpenMP runtime: beside libgomp it livelocks or runs at one core's
speed. `PF_FJBLAS=1` arranges that (torch's `libgomp.so.1` interface served by
Fujitsu's `libfjomphk.so`, Fujitsu's documented threaded runtime preloaded);
then load Fujitsu's BLAS yourself, e.g. a small C wrapper linked with
Fujitsu's documented line (`-lfjlapackexsve -lfjomphk -lfjomp -lfj90i -lfj90f
-lfjsrcinfo -lfjcrt -lfjompcrt -lelf`, built on Fugaku with the host gcc) and
called through `ctypes`. Validated with a source-built torch; Fujitsu
documents this use with other compilers as experimental and unsupported.

## Updating PyTorch

Bump `TORCH_VERSION` in the `Dockerfile`. CI rebuilds torch_fugaku against the
new wheel and runs `tests/` (kernel accuracy against FP64, `bmm` and its
gradients, a 2-rank DDP step over MPI) on GitHub's arm64 runner, whose CPU has
SVE, before pushing.

## Source build (experimental)

`source-build/` builds PyTorch 2.11.0 from source with A64FX patches: a
512-bit SVE CPU capability, batch-parallel `bmm` and Arm Optimized Routines
`erf`/`tanh` inside ATen, and a staged allreduce in the MPI backend, with
Open MPI 4.0.1's ABI so it runs on Fujitsu MPI. It is built only on demand
(`source-build` workflow, tag `:source`). It is correct but currently slower
than this image for OmniLearned (11.2 vs 16.0 events/s on one node): the
SVE512 kernels for ops these patches do not cover are slower than the
wheel's NEON ones. Kept for that work and for upstreaming.

## Licenses

torch_fugaku and the scripts: BSD-3-Clause. `torch_fugaku/csrc/fastmath.cpp`
and `aor_erff_table.h` contain Arm Optimized Routines code and data (MIT OR
Apache-2.0 WITH LLVM-exception). PyTorch and NVPL are installed from their
published wheels under their own licenses. Fujitsu's software is not
redistributed: it is bound from the Fugaku installation at run time.
