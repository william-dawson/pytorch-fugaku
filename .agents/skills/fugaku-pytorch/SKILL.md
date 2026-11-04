---
name: fugaku-pytorch
description: Train PyTorch models on Fugaku (A64FX) with the pytorch-fugaku Singularity image - choosing the per-node layout and the parallel route (DDP, DeepSpeed ZeRO-3, or tensor parallel x FSDP2 for 10B+ models), job scripts with pf-setup/pf-run, the environment settings that matter, expected throughput and memory, and troubleshooting. Use whenever someone wants to run, scale or speed up PyTorch training on Fugaku with this image.
---

# Training with the pytorch-fugaku image

The image (`ghcr.io/william-dawson/pytorch-fugaku`) is the stock PyTorch aarch64 wheel plus
**torch_fugaku**, with every optimization measured on Fugaku on by default (`TORCH_FUGAKU=1`):
- A64FX kernels (gelu, tanh, bmm, transposed GEMM, dropout masks).
- The `"fjmpi"` `torch.distributed` backend over Fujitsu MPI, choosing its reduce-scatter and
  allgather algorithms by layout.
- DeepSpeed ZeRO-3 without its per-rank Python loops.

How it was measured, and every dead end: the `fugaku-pytorch-dev` skill.

## 1. Pick the route by model size

| Model | Route | Layout | Entry point |
|---|---|---|---|
| fits one rank (≲ 300M) | DDP (`gloo` process group; gradients go over MPI automatically) | 4 × 12 | your script |
| ~1-3B | DeepSpeed ZeRO-3 on `fjmpi`, fused chunked loss | 4 × 12 from ~16 nodes (memory) | `examples/llama_train.py` |
| 10-30B+ | tensor parallel 4 in the node × FSDP2 across nodes, sequence parallel | 4 × 12 | `examples/tp_train.py` / `torch_fugaku.parallel` |

**Layout = ranks × OpenMP threads per node.**
- Use **4 × 12** (one rank per CMG) whenever memory allows. It does 1.3-1.8x more compute per
  node than 2 × 24 or 1 × 48, and its communication scales best.
- 2 × 24 and 1 × 48 are memory fallbacks: about 14 or 28 GB per rank instead of about 7.
- Below 4 ranks per node, reductions over MPI degrade at scale.

## 2. Job script

```sh
#PJM -L "rscgrp=small" -L "node=16" -L "elapse=01:00:00"
#PJM -L "freq=2000,eco_state=0"          # normal mode: GEMMs ~1.4x faster than boost-eco
#PJM --mpi "max-proc-per-node=4"
#PJM --llio cn-cache-size=384Mi           # torch's code refaults with the 128 MiB default
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -g <group>
export OMP_NUM_THREADS=12 PF_FJBLAS=1     # PF_FJBLAS=1 only at 4 ranks per node
pytorch-fugaku/fugaku/pf-setup 4          # ranks per node
mpiexec -n 64 pytorch-fugaku/fugaku/pf-run pytorch-fugaku.sif python train.py
```

[references/recipes.md](references/recipes.md) has full scripts for each route: DeepSpeed
config, the tensor-parallel launch, binding code and Python packages in with `PF_BIND` and
`PYTHONPATH`, node shapes.

## 3. Settings that matter

- **Initialize with `init_method="fjmpi://"`.** For example:
  `dist.init_process_group("fjmpi", init_method="fjmpi://")`, or
  `deepspeed.init_distributed(dist_backend="fjmpi", init_method="fjmpi://")`.
  - This skips the TCP rendezvous: fjmpi needs only MPI.
  - The default `env://` makes rank 0 serve one socket per rank, and failed from about 1000
    ranks: first the open-file limit, then connection resets past the listen backlog.

- **`TORCH_FUGAKU_RS_ALGO` / `TORCH_FUGAKU_AG_ALGO`.** Leave at `auto` (rings at 4+ ranks per
  node, one all-to-all reduce-scatter otherwise).
- **`TORCH_FUGAKU_COMM_STATS=1`** times collectives by kind (`torch_fugaku._c10d.stats()`); `=2`
  separates waiting for the slowest rank (skew) from transfer.
- **`TORCH_FUGAKU_FAST_ZERO=0`** turns the ZeRO-3 patch off. It is bit-identical and on by
  default.
- **Memory at 4 × 12** is about 7 GB per rank.
  - Use `torch_fugaku.nn.chunked_lm_loss(model)` for large vocabularies.
  - Build ZeRO-3 models with `from_config` under `zero.Init`; see `llama_train.py --zero-init`.
  - Set `MIMALLOC_PURGE_DELAY=0` when memory is tight (the 30B route). Torch's bundled mimalloc
    otherwise keeps freed memory: peak 5.8 → 3.8 GB per rank, for about 7% more step time.
- **DataLoaders:** `num_workers=0`. Memory registered by Fujitsu MPI is not fork-safe.

## 4. Expectations and troubleshooting

[references/expectations.md](references/expectations.md) lists measured throughput, memory and
communication share by model, layout and node count. Compare your run against it.

Common failures:
- **Exit 137, or a bus error in MPI:** a node ran out of memory. Check per-rank RSS against
  free node memory.
- **Every collective slow at fewer than 4 ranks per node:** expected. Use 4 × 12 if memory
  allows.
- **The first step much slower than the rest at 128+ nodes:** a one-time startup cost; time
  steady steps.
- **`PF_FJBLAS=1` hangs at 2+ ranks per node:** use it only at 4 × 12.
- **"Timed out ... 1012/N clients joined", or repeated "Connection reset by peer" from
  TCPStore, at about 1000+ ranks:** use `init_method="fjmpi://"`.
  - If you must keep `env://`, there are two separate limits:
    - **Open files.** PJM caps each process at 1024 open files, and `ulimit` cannot raise that
      cap. Request more with `#PJM -L "proc-openfd=8192"` (up to 1048576); `pf-run` then raises
      the soft limit to it.
    - **Listen backlog.** `somaxconn` is a node setting that jobs cannot change, so stagger the
      ranks' `init_process_group` calls (a random 0.1-3 s sleep).
- **Slow start-up at hundreds of nodes:** every process imports hundreds of Python modules from
  the shared filesystem at once. Fugaku's `llio_transfer <files>` distributes common files
  before the run; RIKEN reports up to 6x faster PyTorch start-up at 1536 processes.
