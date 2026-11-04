# Benchmark comparisons: legacy SIF, new SIF, LLVM build (November 2026)

Three PyTorch stacks exist on Fugaku. **Keep them apart:**

| Stack | What it is | Runs |
|---|---|---|
| **Legacy SIF** (`DeepSpeedFugaku.sif`) | Fugaku-LLM's Megatron-DeepSpeed in a UBI8 image: Fujitsu's PyTorch 1.13 fork, DeepSpeed 0.8.0, Python 3.9 | Megatron-type GPT training only |
| **New SIF** (`pytorch-fugaku-<commit>.sif`, built by this repository) | stock PyTorch 2.11 wheel + `torch_fugaku` (`fjmpi`, A64FX kernels) | DeepSpeed ZeRO, benchkit, `llama_train.py` |
| **LLVM build** (this skill) | cross-compiled PyTorch 2.11 + the fixes in [performance.md](performance.md) | same as the new SIF |

The legacy SIF cannot run the benchkit workload, and the benchkit comparison does not include it.

## Bottom line

- **Run the LLVM build at 4 ranks × 12 threads per node (one rank per CMG), with fast dropout
  (`assets/fastdrop`) and mimalloc.** Wherever 4 × 12 fits, it is the fastest of the three stacks
  on everything measured:
  - **Legacy SIF:** 1.5× faster at GPT-125M and 2.5× faster at GPT-1.3B per node.
  - **New SIF:** 5-10% faster on benchkit's 12- and 16-node rows. With the new SIF also using the
    fast dropout kernel (now in `torch_fugaku`), LLVM is 2% faster at GPT-125M and 15% faster at
    GPT-1.3B.
- **The container is close behind and covers more layouts.** Wide ranks work there, and it needs
  none of the LLVM build's run-time fixes.
- **The LLVM build is unusable with fewer than 4 ranks per node** (2 × 24, 1 × 48): about 40×
  slower, see below.
- **Benchkit at 4 × 12 fits only at 12 and 16 nodes, on both stacks.** There LLVM leads by 5-10%:
  239 and 225 tokens/s per node, against the PR's best of 206. The PR's 4- and 32-node rows need
  wide ranks for memory, so the LLVM build has no entry for them.

## 1. Benchkit PR 343 (`deepspeed-llm`): new SIF vs LLVM build

[RIKEN-RCCS/benchkit#343](https://github.com/RIKEN-RCCS/benchkit/pull/343): Llama 3.2-1B shapes
(1.24B parameters, 128k vocabulary), sequence 2048, FP32, DeepSpeed ZeRO-3, micro-batch 1,
activation checkpointing, `--low-mem`, 10 steps with 3 warmup, `freq=2000,eco_state=0`.
`PF_FJBLAS=1` only on the 4-ranks-per-node row, as in the PR.

Run with `assets/h2h` (`H2H_MODEL=1b H2H_SEQ=2048 H2H_RPN=<ranks per node> H2H_STEPS=10
H2H_WARMUP=3`), cases `container` and `llvm-port+mi`. The FOM is ranks × 2048 / steady step / nodes.
Llama has no dropout, so fast dropout does not apply here.

| Row | PR's published (new SIF) | New SIF, our run | LLVM build | Jobs |
|---|---:|---:|---:|---|
| 4 nodes, 2 × 24 | 161 tokens/s/node | 160 | ~7 (wide-rank failure) | 53480021 / 53480050 |
| 16 nodes, 4 × 12 | 206 | 204 | **225 (+10%)** | 53480066 / 53480083 |
| 32 nodes, 2 × 24 | 143 | 153 | wide-rank failure | 53480097 / 53480114 |

Our new-SIF runs reproduce the PR, so the harness is faithful. At 16 nodes the losses are
identical (12.1675 at step 9).

**4 × 12 across node counts.** The PR's README says each rank pays about 5 GB for the 128k
vocabulary, so it found 4 ranks per node fit only at 16 nodes. Measured:

| Nodes | New SIF, 4 × 12 | LLVM, 4 × 12 | Jobs |
|---:|---:|---:|---|
| 4 | out of memory in step 1 | out of memory in step-0 backward | 53491470 / 53491461 |
| 8 | — | out of memory in step 1 | 53495276 |
| **12** | **227** (36.1 s/step) | **239 (+5%)** (34.2 s/step) | 53496354 / 53495281 |
| 16 | 204 | 225 (+10%) | 53480066 / 53480083 |
| 24 | — | out of memory (bus error) in step 1 | 53495290 |
| 32 | out of memory, per the README | out of memory in step 1 | 53491466 |

Losses at 12 nodes are identical step by step (12.1695 at step 9).
- **Below 12 nodes** each rank's share of the model state is too big.
- **Above 16 nodes** Fujitsu MPI's per-peer memory grows with the rank count.
- **Neither memory-saving MPI setting rescues 32 nodes:**
  `OMPI_MCA_common_tofu_max_fastmode_procs=16` (53495293) and
  `OMPI_MCA_common_tofu_large_recv_buf_size=262144` (53495301) both still ran out of memory in
  step 1. These are the parameters of the MPI user's guide §6.10.

**The best result on this benchmark is the LLVM build at 4 × 12 on 12 nodes: 239 tokens/s per
node,** against the PR's best of 206.

### Wide ranks (2 × 24, 1 × 48) are broken on the LLVM build

**Symptom.**
- About 40× slower: step 0 took 820-1460 s, against 20-28 s for the new SIF.
- Every collective costs about 1.4 s regardless of size; even a tiny broadcast takes 1.6 s.
- All cores spin inside MPI: CPU time is about 24× wall time, while faults and system time are
  low and compute is normal.
- It is not a straggler: every rank shows the same.

Seen at 4, 8 and 32 nodes. One 2-node 2 × 24 run at sequence 512 was fine, and faster than the
new SIF (53486751 vs 53486754). The same configuration at 4 nodes was slow (53488118).

Profile (job 53487028, 4 nodes, 2 × 24, step 0, rank 0):

| Operator | Self CPU, share of the step |
|---|---:|
| `c10d::_allgather_base_` | 643 s, 75% (1.41 s per call) |
| `c10d::_reduce_scatter_base_` | 185 s, 22% |
| `aten::mm` | 9.9 s |

**Nothing tried fixes it** (all 4 nodes, 1 × 48 unless noted; step-0 forward should be about 9 s):

| Change | Step-0 forward / step | Job |
|---|---|---|
| none (mimalloc) | 150 s / 820 s | 53490104 |
| libmpg instead of mimalloc | 140 s (2 × 24: 32 s step 0, then out of memory) | 53491122 / 53488113 |
| `MIMALLOC_PURGE_DECOMMITS=0` | 104 s | 53491135 |
| backend staging off (`TORCH_FUGAKU_STAGE_MB=0`) | 103 s | 53491146 |
| 46 OpenMP threads instead of 48 | 116 s / 538 s | 53491907 |
| `OMP_WAIT_POLICY=PASSIVE` | 88 s / 367 s | 53491922 |

Two things are ruled out:
- **The hardware barrier:** `FLIB_BARRIER` defaults to SOFT.
- **Thread placement:** all threads float over the rank's CPUs, except two helper threads pinned
  to its first CPU, the same as in the healthy 2-node run.

The new SIF has a related warning in its README: `PF_FJBLAS=1`, which puts it on Fujitsu's
OpenMP, deadlocks above one rank per CMG. The LLVM build always runs on Fujitsu's OpenMP. Cause
still open.

## 2. Legacy SIF (Fugaku-LLM Megatron-DeepSpeed) vs the new stacks

### Running the legacy SIF

- **Location.** Image and inputs, as provided, are in `/share/rccs-sdt/a01029/toRA250029/`:
  `DeepSpeedFugaku.sif` (4.1 GB), `DeepSpeedFugaku.tar.gz` (the code bundle with dataset and
  tokenizers) and Fujitsu's PyTorch 1.13 tarball.
- **`/share` is not mounted on compute nodes.** Copy the SIF and extract the bundle into a data
  volume.
- **Use `singularity run`.** The runscript activates `/local/fcc/inst/venv`; with `exec`, `torch`
  is not importable.
- **Bind the host's Fujitsu runtime.** The image expects tcsds-1.2.36 at
  `/opt/FJSVxtclanga/tcsds-1.2.36/lib64`; without a bind, python fails on
  `libfjlapackexsve.so.1`. Bind `/opt/FJSVxtclanga` and the host's interconnect libraries exactly
  as `fugaku/pf-run` does, and add `/hostlib` to `LD_LIBRARY_PATH`. Host `mpiexec` (tcsds 1.2.43)
  then runs ranks over the image's 1.2.36 MPI, including tensor parallelism across nodes.
- **Run the bundle's own `pretrain_gpt.py`.** The image ships only upstream Megatron-DeepSpeed
  under `/usr/local`. Run it from the bundle's directory
  (`bash -c "cd $D && exec python $D/pretrain_gpt.py ..."`; `--pwd` was ignored). The bundle has
  a prebuilt Python 3.9 data helper, the tokenized codeparrot dataset, and the tokenizer
  `JapaneseSentencePiece code20k_en40k_ja60k.ver2.1` (vocabulary padded to 101,120).
- **No tcmalloc.** Their `LD_PRELOAD=libtcmalloc.so` cannot load: `libunwind.so.8` is in neither
  the image nor the compute nodes, and preloading it anyway breaks even the runscript's shell.
  Their 2.2 GHz runs are not reproduced either.
- **Per-step profiler.** The bundle's `megatron/training.py` profiles every training step and
  prints a 100-row table, a debug leftover. `assets/legacy/training-profiler-optin.patch` makes it
  opt-in (`H2H_LEGACY_PROFILE=1`).

`assets/legacy/smoke.sh` is the first working run. `assets/legacy/bench.sh` is the benchmark
(`L_MODEL`, `L_RPN`, `L_MBS`, `L_SEQ`, `L_ITERS`, `L_TP`, `L_DROPOUT`, `L_MALLOC`).

### Which configurations are comparable

Fugaku-LLM's notes (`experiments/124m_13b_benchmarking/*.md` in the bundle) include only two sizes
that ran data-parallel only:

| Model | Their layout | Best per-node throughput in their notes |
|---|---|---|
| GPT 125M (12 × 768) | 4 ranks × 12 per node | ≈2,900 tokens/s (PyTorch 1.10, 2.2 GHz, tcmalloc, global batch 1024) |
| GPT 1.3B (24 × 2048) | 1 rank × 48, 28 GB (PyTorch 1.7); tensor-parallel 4 over 4 nodes, 16.6 GB (PyTorch 1.13) | ≈104 tokens/s |

Larger models need tensor parallelism 20-40 or pipeline parallelism 12-48; the 30B's 48-node run
ran out of memory. Their single-node figures accumulate 256-1024 micro-batches per optimizer
step.

### Results

FP32, sequence 1024, activation checkpointing, `freq=2000,eco_state=0`, dropout 0.1 unless
stated.
- **Legacy:** Megatron through `bench.sh`, one sequence per rank per step, no accumulation.
- **New stacks:** `llama_train.py --model gpt-125m|gpt-1.3b` (GPT-2 architecture, vocabulary
  50,304) through `assets/h2h`.
- Tokens/s per node = sequences per step × 1024 / steady step / nodes.

| GPT-125M, 1 node | Layout | Step | Tokens/s/node | Job |
|---|---|---:|---:|---|
| Legacy | 4 × 12 data parallel, micro-batch 4 | 14.5 s | 1,130 | 53487626 |
| Legacy, dropout 0 | same | 12.0 s | 1,365 | 53492129 |
| New SIF | 4 × 12 ZeRO-0, micro-batch 4 | 38.7 s | 423 | 53487642 |
| LLVM | same | 32.1 s | 511 | 53487645 |
| **LLVM + fast dropout** | same | **9.54 s** | **1,717** | 53493468 |
| LLVM, dropout 0 | same | 8.07 s | 2,030 | 53488227 |
| **New SIF + fast dropout** (torch_fugaku `bernoulli`) | same | **9.77 s** | **1,677** | 53498044 |
| LLVM + fast dropout, same time as the row above | same | 9.55 s | 1,716 | 53498059 |

At 4 nodes the 125M per-node rates are the same within 2% (legacy 53487631, new SIF 53487649,
LLVM 53487654).

| GPT-1.3B, 4 nodes | Layout | Step | Tokens/s/node | Job |
|---|---|---:|---:|---|
| Legacy | tensor-parallel 4, 1 rank × 48 per node | 3.68 s / sequence | 70 | 53492139 |
| Legacy, dropout 0 | same | 3.52 s | 73 | 53492149 |
| New SIF | 4 × 12 ZeRO-3 | 48.7 s / 16 sequences | 84 | 53492186 |
| LLVM | same | 40.2 s | 102 | 53492160 |
| **LLVM + fast dropout** | same | **23.4 s** | **175** | 53493472 |
| LLVM, dropout 0 | same | 21.3 s | 193 | 53492175 |
| **New SIF + fast dropout** (torch_fugaku `bernoulli`) | same | **25.5 s** | **161** | 53498082 |
| LLVM + fast dropout, same time as the row above | same | 22.2 s | 185 | 53498089 |

**With fast dropout on both new stacks** (`torch_fugaku/csrc/bernoulli.cpp`, the same algorithm as
`assets/fastdrop`), the masks are bit-identical between them and so are the losses, step by step.
- **GPT-125M:** the new SIF nearly matches LLVM, 1,677 vs 1,716 tokens/s per node (+2% LLVM).
- **GPT-1.3B:** LLVM leads by 15% (185 vs 161).
- **Against the legacy SIF:** both are well ahead, 1.5× at 125M and 2.3-2.6× at 1.3B.
- **Effect on the new SIF itself:** the port made it 4.0× faster at 125M (423 → 1,677) and 1.9× at
  1.3B (84 → 161).

**Plain data-parallel 1.3B fits nowhere** in a 32 GB node: 1 rank × 48 ran out of memory on the
legacy SIF (glibc and mimalloc), the new SIF and LLVM, on 1 and 4 nodes. Jobs:
53487634 / 53487639 / 53487917 / 53487920 / 53487662 / 53487666 / 53487669 / 53487674.

The legacy's 125M also ran out of memory with mimalloc (53487923), after 10.9 s iterations.

Caveats:
- **Batch per step.** Legacy processes 1-4 sequences per optimizer step here, the new stacks 4-16.
  Larger global batches favour Megatron, as its notes show.
- **Training code.** Megatron and HF-GPT2 differ; for example, Megatron fuses bias + GELU.
- **Vocabulary.** The legacy's 101k vocabulary is twice GPT-2's.

### Why the legacy SIF won at 125M: dropout, not the activation function

Profile of one GPT-125M step, rank 0, self CPU (jobs 53488190 / 53488202 / 53488215):

| | Legacy (Fujitsu PyTorch 1.13) | LLVM | New SIF |
|---|---:|---:|---:|
| `aten::bernoulli_` (dropout masks, 73 calls) | **0.69 s** | **22.8 s (71%)** | **33.3 s** |
| GELU | 0.76 s `tanh`, inside fused `bias_gelu` | 0.64 s `tanh` | 0.08 s (fast kernel) |
| `mm` + `addmm` + `bmm` | 4.7 s | 3.95 s | 4.25 s |

Stock PyTorch's CPU `bernoulli_` is scalar on aarch64; Fujitsu's 1.13 fork does it about 33×
faster. GELU differs by under 2% of the step. Matmuls are comparable, with LLVM slightly ahead.
`assets/fastdrop` fixes the dropout cost for the LLVM build; see [performance.md](performance.md).
The new SIF gets the same kernel as `torch_fugaku`'s `bernoulli` override. Built with the image's
g++ 12 (generic aarch64), it runs at 6.3 ms against 312 ms stock for 12.6M elements (job 53497747).
The SVE-tuned LLVM build runs it at 1.85 ms.
