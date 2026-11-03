# Benchmark comparisons: legacy SIF, new SIF, LLVM build (November 2026, in progress)

Three PyTorch stacks exist on Fugaku. **Keep them apart:**

| Stack | What it is | Runs |
|---|---|---|
| **Legacy SIF** (`DeepSpeedFugaku.sif`) | Fugaku-LLM's Megatron-DeepSpeed in a UBI8 image: Fujitsu's PyTorch 1.13 fork, DeepSpeed 0.8.0, Python 3.9 | Megatron-type GPT training only |
| **New SIF** (`pytorch-fugaku-<commit>.sif`, built by this repository) | stock PyTorch 2.11 wheel + `torch_fugaku` (`fjmpi`, A64FX kernels) | DeepSpeed ZeRO, benchkit, `llama_train.py` |
| **LLVM build** (this skill) | cross-compiled PyTorch 2.11 + the fixes in [performance.md](performance.md) | same as the new SIF |

The legacy SIF cannot run the benchkit workload, and the benchkit comparison does not include it.

## 1. Benchkit PR 343 (`deepspeed-llm`): new SIF vs LLVM build

[RIKEN-RCCS/benchkit#343](https://github.com/RIKEN-RCCS/benchkit/pull/343): Llama 3.2-1B shapes
(1.24B parameters, 128k vocabulary), sequence 2048, FP32, DeepSpeed ZeRO-3, micro-batch 1,
activation checkpointing, `--low-mem`, 10 steps with 3 warmup, `freq=2000,eco_state=0`.
`PF_FJBLAS=1` only on the 4-ranks-per-node row, as in the PR.

Run with `assets/h2h` (`H2H_MODEL=1b H2H_SEQ=2048 H2H_RPN=<ranks per node> H2H_STEPS=10
H2H_WARMUP=3`), cases `container` and `llvm-port+mi`. The FOM is ranks × 2048 / steady step / nodes.

| Row | PR's published result (new SIF) | New SIF, our run | LLVM build | Jobs |
|---|---:|---:|---:|---|
| 4 nodes, 2 ranks × 24 threads | 161 tokens/s/node | 160 | **~7: broken, see below** | 53480021 / 53480050 |
| 16 nodes, 4 ranks × 12 threads | 206 | 204 | **225 (+10%)** | 53480066 / 53480083 |
| 32 nodes, 2 ranks × 24 threads | 143 | 153 | **broken** | 53480097 / 53480114 |

Our new-SIF runs reproduce the PR's figures, so the harness is faithful. At 4 × 12 the LLVM build
is 10% faster with identical losses (12.1675 at step 9 on both).

**Open problem: LLVM at 2 ranks × 24 threads with sequence 2048.**
- The runs are about 40× slower. Step 0 took 1194 s on 4 nodes and 1462 s on 32 nodes, against about 28 s for the new SIF. One forward took 352 s.
- It is not a hang: steps complete, and both jobs hit the 32-minute case timeout (exit 124).
- It is specific to that combination:
  - At sequence 512 on 2 nodes, 2 × 24, the LLVM build is **faster** than the new SIF: 11.7 vs 13.6 s/step, identical loss (jobs 53486751 / 53486754).
  - In that run Fujitsu's BLAS made matmuls faster (2.95 vs 4.01 s self-CPU), and flash-attention backward was 0.43 vs 0.36 s. Even at 16× attention cost, that is seconds, not the 1000+ seconds observed.
- An operator profile of the failing configuration is job 53487028.
- The new SIF does not use Fujitsu's BLAS at 2 ranks per node (README: `PF_FJBLAS=1` deadlocks above one rank per CMG). The LLVM build always does.

## 2. Legacy SIF (Fugaku-LLM Megatron-DeepSpeed)

### Where it is and what it needs

- Image and inputs, as provided: `/share/rccs-sdt/a01029/toRA250029/`. It holds `DeepSpeedFugaku.sif` (4.1 GB), `DeepSpeedFugaku.tar.gz` (the code bundle with dataset and tokenizers) and Fujitsu's PyTorch 1.13 tarball.
- **`/share` is not mounted on compute nodes.** Copy the SIF and extract the bundle into a data volume.
- The runscript activates `/local/fcc/inst/venv`. Use **`singularity run`**: with `exec`, `torch` is not importable.
- **The image expects the host's Fujitsu runtime** at `/opt/FJSVxtclanga/tcsds-1.2.36/lib64` (its `LD_LIBRARY_PATH`). Without a bind, python fails on `libfjlapackexsve.so.1`. Bind `/opt/FJSVxtclanga` and the host's interconnect libraries exactly as `fugaku/pf-run` does (pf-setup's hostlib, `/var/opt/FJSVtcs`, `/run`, ...), and add `/hostlib` to `LD_LIBRARY_PATH`. Host `mpiexec` (tcsds 1.2.43) then runs 4 ranks over the image's 1.2.36 MPI.
- **The image does not contain Fugaku-LLM's `pretrain_gpt.py`**, only upstream Megatron-DeepSpeed under `/usr/local`. Run the bundle's copy from its own directory (`bash -c "cd $D && exec python $D/pretrain_gpt.py ..."`; `--pwd` was ignored). The bundle ships a prebuilt Python 3.9 data helper and the tokenized codeparrot dataset; its tokenizer is `JapaneseSentencePiece code20k_en40k_ja60k.ver2.1`.
- **Their `LD_PRELOAD=libtcmalloc.so` cannot be reproduced.** It needs `libunwind.so.8`, which is in neither the image nor the compute nodes. Preloading it anyway breaks even the runscript's shell. Their freq=2200 runs are also not reproduced (we use 2000, normal mode).
- **The bundle's `megatron/training.py` profiles every training step** and prints a 100-row table, a debug leftover. `assets/legacy/training-profiler-optin.patch` makes it opt-in (`H2H_LEGACY_PROFILE=1`); apply it to the extracted copy.

`assets/legacy/smoke.sh` is the first working run. `assets/legacy/bench.sh` is the parameterized
benchmark (`L_MODEL`, `L_RPN`, `L_MBS`, `L_SEQ`, `L_ITERS`).

### Which configurations are comparable

Fugaku-LLM's own notes (`experiments/124m_13b_benchmarking/*.md` in the bundle) include only two
sizes that run **purely data-parallel** (tensor- and pipeline-parallel size 1). Larger models need
tensor or pipeline parallelism, which would compare parallelization strategies rather than stacks.

| Model | Shape | Their DP-only layout | Best per-node throughput in their notes |
|---|---|---|---|
| GPT 125M | 12 layers, 768 hidden, 12 heads | 1 node 4 ranks × 12, up to 512 nodes | ≈2,900 tokens/s (1 node, PyTorch 1.10, 2.2 GHz, tcmalloc, global batch 1024) |
| GPT 1.3B | 24 layers, 2048 hidden, 32 heads, checkpointing | 1 rank × 48 per node (28 GB), 1 and 64 nodes | ≈105 tokens/s (1 node, PyTorch 1.7) |

Throughput here is global batch × sequence ÷ seconds per iteration ÷ nodes. Their single-node
figures accumulate 256-1024 micro-batches per optimizer step. At 512 nodes (125M) and 64 nodes
(1.3B) the same notes give about 640 and 78 tokens/s per node.
- 13B needs tensor parallelism 20-40 or pipeline parallelism 12.
- 30B needs pipeline parallelism 48 plus tensor parallelism 4-8. Its smallest run, 48 nodes, ran out of memory.

### First measurements (smoke, job 53487014; profiler still on, no tcmalloc)

GPT 125M, sequence 1024, micro-batch 1, no accumulation:
- 1 rank × 48 threads: about 10.8 s per iteration, about 95 tokens/s.
- 4 ranks × 12 threads: about 13.2 s per 4-sample iteration, about **310 tokens/s per node**.
- Loss went 11.0 → 10.5 over 6 iterations.

These are not comparable to the notes' accumulated figures.

### Comparison in progress

Same configurations on all three stacks, data-parallel only, sequence 1024, activation
checkpointing, FP32, no gradient accumulation, `freq=2000,eco_state=0`. The new SIF and LLVM run
`llama_train.py --model gpt-125m|gpt-1.3b` (GPT-2 architecture, vocabulary 50304) with ZeRO-0
(`H2H_ZERO=0`). The legacy side runs Megatron through `bench.sh`.

| Model | Layout | Micro-batch | Nodes |
|---|---|---|---|
| GPT 125M | 4 ranks × 12 | 4 | 1, 4 |
| GPT 1.3B | 1 rank × 48 | 1 | 1, 4 |

Jobs 53487626-53487674. Megatron and HF-GPT2 are different training codes, so the result compares
the best stack each environment offers, not one script.
