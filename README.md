# pytorch-fugaku

PyTorch for [Fugaku](https://www.r-ccs.riken.jp/en/fugaku/) (Fujitsu A64FX), as a
container image: the official PyTorch aarch64 wheel plus **torch_fugaku**, a
small package that makes it fast on A64FX and runs distributed training over
Fujitsu MPI.

```
ghcr.io/william-dawson/pytorch-fugaku:latest     # linux/arm64
```

## Run on Fugaku

On a login node, pull the image and get the two launch scripts. Keep
Singularity's cache on a data volume; the image does not fit the home quota.

```sh
export SINGULARITY_CACHEDIR=/vol0X0X/data/<group>/.singularity
singularity pull --arch arm64 pytorch-fugaku.sif docker://ghcr.io/william-dawson/pytorch-fugaku:latest
git clone https://github.com/william-dawson/pytorch-fugaku.git
```

A job, one rank per CMG (4 per node, 12 threads each):

```sh
#PJM -L node=4
#PJM -L freq=2000,eco_state=0
#PJM --mpi max-proc-per-node=4
export OMP_NUM_THREADS=12
pytorch-fugaku/fugaku/pf-setup
mpiexec -n 16 pytorch-fugaku/fugaku/pf-run pytorch-fugaku.sif python train.py
```

`train.py` is ordinary PyTorch: `init_process_group("gloo")` and
`DistributedDataParallel`. The two scripts do the Fugaku-specific part:

1. **`pf-setup`**, once per job: collects the compute node's interconnect
   libraries that Fujitsu MPI needs, and writes the process-group settings
   (`WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `GLOO_SOCKET_IFNAME`).
2. **`pf-run`**, once per rank under Fujitsu `mpiexec`: starts the container
   with Fujitsu MPI bound in, and sets `RANK` and `LOCAL_RANK`.

Also:

- `PF_BIND` binds your data and code into the container (comma-separated).
- Use `num_workers=0` in DataLoaders: memory registered by Fujitsu MPI is not
  fork-safe.
- `-L freq=2000,eco_state=0` selects normal mode, in which matrix
  multiplications run ~1.4x faster than in the default boost-eco mode.

## What torch_fugaku does

It applies itself when torch is imported (`TORCH_FUGAKU=1` in the image):

- **A64FX kernels**: `bmm` threaded over the batch, `mm`/`addmm` with a
  transposed right operand (`nn.Linear`'s forward) through a fast transpose and
  the plain GEMM, SVE `gelu`,
  `gelu_backward` and `tanh` from Arm Optimized Routines, and the whole-tensor
  2-norm that gradient clipping computes (threaded, double accumulation),
  registered as CPU kernels with `TORCH_LIBRARY_IMPL`.
- **Gradients over Fujitsu MPI**: a DDP communication hook sums each gradient
  bucket with `MPI_Allreduce` over Tofu. The process group stays gloo; only
  the gradient transport changes.
- **A `torch.distributed` backend over Fujitsu MPI**, `"fjmpi"`, for code that
  calls collectives itself (DeepSpeed's ZeRO, FSDP):
  `init_process_group("fjmpi")`. All-reduce, broadcast, reduce, all-gather,
  reduce-scatter, all-to-all, barrier and subgroups, on CPU tensors.
  Reduce-scatter runs as an all-to-all and a local sum: Fujitsu MPI's own is
  several times slower. `TORCH_FUGAKU_STAGE_MB` sets the staging buffers
  (default 8; 0 sends tensors directly).
- **Settings**: oneDNN off, `OMP_WAIT_POLICY=ACTIVE`.

The same, explicitly (with `TORCH_FUGAKU=0`):

```python
import torch_fugaku
torch_fugaku.enable()              # kernels and settings
model = torch_fugaku.ddp(model)    # DistributedDataParallel with gradients over MPI
torch_fugaku.register_backend()    # "fjmpi" for torch.distributed
torch_fugaku.status()              # what is active
```

Switches: `TORCH_FUGAKU_KERNELS=0`, `TORCH_FUGAKU_ONEDNN=1`,
`TORCH_FUGAKU_MPI=0|1|auto` (auto: on when mpiexec started the process).

## Running DeepSpeed

ZeRO (and FSDP) call `torch.distributed` collectives directly, so point DeepSpeed at the
`"fjmpi"` backend instead of initialising a process group yourself:

```python
import torch_fugaku
torch_fugaku.register_backend()
import deepspeed
deepspeed.init_distributed(dist_backend="fjmpi")
```

On the CPU accelerator (`DS_ACCELERATOR=cpu`), DeepSpeed JIT-compiles a shared-memory comm op
with `ninja`/`gcc` the first time it's used; point `TORCH_EXTENSIONS_DIR` at a writable path
(reusing it across runs is fine -- the build is keyed on torch/DeepSpeed's ABI, not your model).
`num_workers=0` and `PF_BIND` apply the same as any other job (above).

**`PF_FJBLAS=1` is not safe with more than one rank per node.** Two failure modes, both
reproducible: at 2+ ranks per node, DeepSpeed's intra-node shared-memory comm setup deadlocks
after it reports the comm op built successfully, before the first step; at exactly 1 rank per
node but more than one physical node, the process-group rendezvous itself deadlocks instead.
Confirmed by A/B: identical job, only `PF_FJBLAS` toggled -- with it, hangs through the wall-time
limit at the point above; without it, completes normally. Only exactly one rank per CMG
(4 ranks x 12 threads per node) is validated safe with `PF_FJBLAS=1` today.

## Performance

[OmniLearned](https://github.com/ViniciusMikuni/OmniLearned) pretraining (PET v2
medium, FP32, batch 8 per rank, 4 ranks per node, normal mode), events/s:

| nodes | pytorch-fugaku |
|---:|---:|
| 1 | 16.8 |
| 4 | 55.4 |
| 12 | 150.8 |

The stock wheel, run plainly in the same container, reaches 0.49 events/s on
one node.

DeepSpeed (0.19.8) training a GPT-2-small-sized model (163M parameters, FP32,
2 sequences of 1024 per rank, 4 ranks per node, normal mode), seconds per step:

| nodes | ZeRO-0, gloo | ZeRO-0, fjmpi | ZeRO-3, gloo | ZeRO-3, fjmpi |
|---:|---:|---:|---:|---:|
| 1 | 4.1 | 3.3 | 12.2 | 4.1 |
| 2 | 12.2 | 3.5 | 24.0 | 4.5 |
| 4 | 13.9 | 3.5 | 31.6 | 4.4 |

Losses are identical between the two backends.

Kernels on one CMG, 16 M float32 elements:

| op | stock wheel | pytorch-fugaku |
|---|---:|---:|
| `gelu` | 17.1 ms | 5.3 ms |
| `gelu_backward` | 29.7 ms | 3.9 ms |
| `tanh` | 43.5 ms | 4.0 ms |
| `bmm`, 4096 x 32x32x32 | 111 ms | 7.2 ms |
| `linalg_vector_norm` (gradient clipping), 163 M float32 in 149 tensors | 197 ms | 23 ms |
| `X . W^T`, 2048x8192 by (2048x8192)^T (Llama `down_proj`) | 170 GF/s | 696 GF/s |

## Fujitsu's linear algebra (optional)

`PF_FJBLAS=1` runs each rank on Fujitsu's OpenMP runtime, with thread binding off (it would put
every thread on one core), and torch_fugaku then sends float32 GEMMs to Fujitsu's threaded BLAS
(`libfjlapackexsve`, loaded from the bound Fujitsu installation at run time; nothing of
Fujitsu's is in the image). On one CMG it runs Llama-sized GEMMs at 1,000-1,200 GF/s against
NVPL's 600-1,000. It pays with one rank per CMG (4 ranks x 12 threads per node): Llama-1B under
DeepSpeed ZeRO-3 at 16 nodes went from 49.7 to 41.4 s per step. **With fewer, wider ranks it is
not just a smaller win -- see the deadlock warning under "Running DeepSpeed."**

## Build and update

`Dockerfile` builds on any arm64 Linux host. CI builds it on GitHub's arm64
runner, tests it (kernel accuracy against FP64, `bmm` and its gradients, a
2-rank DDP step over MPI) and pushes it. To move to a new PyTorch, change
`TORCH_VERSION` in the `Dockerfile`.

## Licenses

BSD-3-Clause. `torch_fugaku/csrc/fastmath.cpp` and `aor_erff_table.h` contain
Arm Optimized Routines code and data (MIT OR Apache-2.0 WITH LLVM-exception).
PyTorch and NVPL come from their published wheels. Fujitsu software is not
redistributed; it is bound from the Fugaku installation at run time.
