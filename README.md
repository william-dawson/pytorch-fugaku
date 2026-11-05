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

For a pinned, checkable file, pull the SIF that CI publishes for each image instead
(`sif-<short commit>`, with its SHA-256 in that run's summary of the `sif` workflow). Converting
the image yourself gives a working file but a different SHA-256 every time.

```sh
singularity pull --disable-cache pytorch-fugaku.sif oras://ghcr.io/william-dawson/pytorch-fugaku:sif-<short commit>
sha256sum pytorch-fugaku.sif
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

**Agent skills.** [`fugaku-pytorch`](.agents/skills/fugaku-pytorch/SKILL.md) covers training well with
the image: which layout and parallel route for which model size, job scripts, settings, and
expected throughput and memory. [`fugaku-pytorch-dev`](.agents/skills/fugaku-pytorch-dev/SKILL.md)
is the developer record: every experiment and dead end behind the image, the benchmark harness,
and the LLVM source-build route.

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
  Reduce-scatter and all-gather pick their algorithms (`TORCH_FUGAKU_RS_ALGO`,
  `TORCH_FUGAKU_AG_ALGO`, default `auto`) from the job's ranks per node: rings
  at 4+ (all-gather only in groups with one rank per node), one unstaged
  all-to-all for reduce-scatter otherwise. Fujitsu MPI's own is several times
  slower. `TORCH_FUGAKU_STAGE_MB` sets the staging buffers (default 8; 0 sends
  tensors directly). `TORCH_FUGAKU_COMM_STATS=1` times collectives by kind
  (`torch_fugaku._c10d.stats()`); `=2` adds a barrier first to separate skew.
- **DeepSpeed ZeRO-3 without per-rank Python loops** (`torch_fugaku.deepspeed`), applied when
  ZeRO-3 is imported: DeepSpeed's coalesced reduce-scatter and all-gather bookkeeping cost
  O(ranks × tensors) per step (4.3 s at 512 ranks for a 1B model); the patch is one strided copy
  per tensor, with bit-identical results.
- **Settings**: oneDNN off, `OMP_WAIT_POLICY=ACTIVE`. When memory is tight, set
  `MIMALLOC_PURGE_DELAY=0`: torch's CPU allocator is a bundled mimalloc whose arenas otherwise keep
  gigabytes of freed memory (peak 5.8 → 3.8 GB per rank, at ~7% speed).

For large models, two opt-in modules: `torch_fugaku.nn.chunked_lm_loss(model)` (a causal-LM loss
that never holds the [tokens, vocab] logits) and `torch_fugaku.parallel` (tensor parallel inside
the node × FSDP2 across nodes for Llama). `examples/` has a DeepSpeed ZeRO-3 trainer and a
tensor-parallel one; see "Large models".

The same, explicitly (with `TORCH_FUGAKU=0`):

```python
import torch_fugaku
torch_fugaku.enable()              # kernels and settings
model = torch_fugaku.ddp(model)    # DistributedDataParallel with gradients over MPI
torch_fugaku.register_backend()    # "fjmpi" for torch.distributed
torch_fugaku.status()              # what is active
```

Switches: `TORCH_FUGAKU_KERNELS=0`, `TORCH_FUGAKU_ONEDNN=1`,
`TORCH_FUGAKU_MPI=0|1|auto` (auto: on when mpiexec started the process),
`TORCH_FUGAKU_FAST_ZERO=0`.

## Running DeepSpeed

DeepSpeed 0.19.8, transformers 4.57.6 and accelerate 1.15.0 are in the image (`DS_ACCELERATOR=cpu`).
ZeRO (and FSDP) call `torch.distributed` collectives directly, so point DeepSpeed at the
`"fjmpi"` backend instead of initialising a process group yourself:

```python
import torch_fugaku
torch_fugaku.register_backend()
import deepspeed
deepspeed.init_distributed(dist_backend="fjmpi", init_method="fjmpi://")
```

`init_method="fjmpi://"` takes rank and size from the MPI launcher, with no TCP rendezvous. The
default `env://` makes rank 0 serve one socket per rank, which failed on Fugaku from about 1000
ranks.

torch_fugaku marks DeepSpeed's C++ ops unavailable when DeepSpeed is imported, so DeepSpeed does
not try to JIT-build them (the image has no ninja), and its collectives go through fjmpi. Every
recorded measurement ran this way. `TORCH_FUGAKU_DS_OPS=1` restores DeepSpeed's default; it then
JIT-compiles a shared-memory comm op with `ninja`/`gcc` and needs a writable
`TORCH_EXTENSIONS_DIR`. `num_workers=0` and `PF_BIND` apply the same as any other job (above).

**`PF_FJBLAS=1` is not safe with more than one rank per node** when DeepSpeed's shared-memory op
is in use (`TORCH_FUGAKU_DS_OPS=1`); with the default (ops off), 4 × 12 with `PF_FJBLAS=1` is
what every measurement used. Two failure modes, both
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
| `bernoulli_(0.9)` (dropout's mask), 12.6 M float32 | 312 ms | 6.3 ms |

The `bernoulli_` override (a keyed counter hash, seeded from torch's generator) makes GPT-style
training with dropout about 4x faster: GPT-2-sized (125M) at 4 ranks x 12 threads went from 38.7
to 9.8 s per step. Its random stream differs from the stock wheel's, so losses are not
bit-identical to runs without torch_fugaku.
| `X . W^T`, 2048x8192 by (2048x8192)^T (Llama `down_proj`) | 170 GF/s | 696 GF/s |

## Large models

Benchkit Llama-1B (DeepSpeed ZeRO-3, seq 2048), tokens/s per node:

| nodes | 1 × 48 | 2 × 24 | 4 × 12 |
|---:|---:|---:|---:|
| 16 | 97 | 129 | 184 |
| 128 | 79 | 108 | 173 |

At 4 × 12 (one rank per CMG), it needs `--loss-chunk` (`torch_fugaku.nn`) to fit.

A 30B model (the Fugaku-LLM Megatron SIF's GPT-30B shape) does not fit ZeRO-3 at 4 ranks per
node. With tensor parallel 4 inside each node × FSDP2 across nodes (`examples/tp_train.py`), it
trains at 6.0, 5.7, 7.6 and 6.6 tokens/s per node on 64, 128, 256 and 576 nodes. The legacy
Megatron SIF does about 6.0 on 576 nodes. Details, including memory and what limits scaling, are in the `fugaku-pytorch`
skill.

## Fujitsu's linear algebra (optional)

`PF_FJBLAS=1` runs each rank on Fujitsu's OpenMP runtime, with thread binding off (it would put
every thread on one core), and torch_fugaku then sends float32 GEMMs to Fujitsu's threaded BLAS
(`libfjlapackexsve`, loaded from the bound Fujitsu installation at run time; nothing of
Fujitsu's is in the image). On one CMG it runs Llama-sized GEMMs at 1,000-1,200 GF/s against
NVPL's 600-1,000. It pays with one rank per CMG (4 ranks x 12 threads per node): Llama-1B under
DeepSpeed ZeRO-3 at 16 nodes went from 49.7 to 41.4 s per step. **With fewer, wider ranks it is
not just a smaller win -- see the deadlock warning under "Running DeepSpeed."**

## LLVM cross-compilation (source route)

A separate source-build route cross-compiles PyTorch 2.11.0 on Fugaku's x86 login
nodes for A64FX with LLVM 22.1.0, Fujitsu threaded BLAS/LAPACK, Fujitsu OpenMP and
PyTorch's built-in `"mpi"` backend. It produces a Python 3.11 wheel independently
of the container and `torch_fugaku` extension described above.

The build first prepares native code generators on x86, then cross-compiles
LibTorch and its MPI backend. After validating the core on compute nodes, it
builds the Python bindings against AArch64 Python headers and packages the
wheel. The resulting wheel has passed two-node tests for matrix multiplication,
LAPACK, autograd, NumPy interoperability, OpenMP and blocking/asynchronous MPI
operations. Historical one-node GEMM measurements cover 1, 2 and 4 MPI ranks; those
benchmarks have not yet been repeated for the allocator rebuild.

The fresh source build explicitly disables PyTorch’s AArch64 mimalloc override,
links Fugaku’s large-page library, and supplies a linked non-PIE Python runner.
Two-node tests verified actual 2 MiB tensor pages and seven asynchronous MPI
cases, with application `LD_PRELOAD` and `LD_LIBRARY_PATH` cleared. See the
[large-page build and validation notes](.agents/skills/fugaku-pytorch-dev/references/large-pages.md).

**Against the container**, the cross-build reaches the container's speed, and beats it by
5-25% on small DeepSpeed runs, but only with run-time fixes (LLIO cache size, the container's
backend, a preloaded allocator). It runs well only at 4 ranks per node, and at 32-128 nodes it was
slower than the container (146 vs 175 and 126 vs 164 tokens/s/node for Llama-1B). The container is
the recommended route; the developer skill has the comparisons.

The [developer skill](.agents/skills/fugaku-pytorch-dev/SKILL.md)
contains pinned inputs, the build scripts and patches, a fresh-workspace
reproduction procedure, troubleshooting, and recorded two-node C++/Python and
asynchronous MPI tests. It also records 1/2/4-rank GEMM results and remaining
performance/coverage limitations. Built wheels and commercial Fujitsu libraries
are not stored in this repository.

## Build and update

`Dockerfile` builds on any arm64 Linux host. CI builds it on GitHub's arm64
runner, tests it (kernel accuracy against FP64, `bmm` and its gradients, the chunked loss, a
2-rank DDP step over MPI, the `fjmpi` backend on 2 and 4 ranks with default and ring algorithms,
and tensor parallel × FSDP2 against an unsharded copy) and pushes it. To move to a new PyTorch, change
`TORCH_VERSION` in the `Dockerfile`.

## Licenses

BSD-3-Clause. `torch_fugaku/csrc/fastmath.cpp` and `aor_erff_table.h` contain
Arm Optimized Routines code and data (MIT OR Apache-2.0 WITH LLVM-exception).
PyTorch and NVPL come from their published wheels. Fujitsu software is not
redistributed; it is bound from the Fugaku installation at run time.
