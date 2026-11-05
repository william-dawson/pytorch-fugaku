# The same FSDP2 training on Fugaku, RIKYU, DGX Spark and Genoa (November 2026)

benchkit's `fsdp-llm` (RIKEN-RCCS/benchkit PR #343) runs one `llm_train.py` on four machines:
Llama-architecture models on synthetic tokens, seq 2048, AdamW, activation checkpointing,
5 steps with 2 excluded. The trainer carries its own tensor-parallel plan (the same as
`torch_fugaku.parallel`), since the GPU machines have no torch_fugaku.

- **30B:** the GPT-30B configuration from DeepSpeedFugaku's job scripts, as a Llama
  (28.2B parameters). Tensor parallel = ranks per node, FSDP2 across nodes.
- **1B:** Llama 3.2-1B shape, untied (1.5B parameters).

## Results (2026-11-05, every row through benchkit's `scripts/test_submit.sh`)

Tokens/s per node:

| machine | layout | 1B | 30B |
|---|---|---:|---:|
| RIKYU, 4 x B200, bf16 | 1 rank per GPU, 16 CPUs each | 240,640-243,083 (16 seq/GPU, FSDP2 alone) | 12,368-12,586 (64 seq per TP group) |
| DGX Spark, GB10, bf16 | 1 rank | 3,537 (16 seq) | does not fit (~450 GB of state) |
| Genoa, 2 x EPYC 9684X, fp32 | 16 x 6, gloo | 384 (4 seq/rank, FSDP2 alone) | 13.1 (fits in 768 GB; 624 s steps) |
| Fugaku, A64FX, fp32 | 4 x 12, fjmpi | 176-178 at 4 / 16 nodes (TP 4, 4 seq per group) | 7.5-9.0 at 64-256 nodes |

## Findings

| Finding | Evidence | Status |
|---|---|---|
| **Tensor parallel of size 1 is not free.** With `--tp 1` the DTensor plan still wrapped every layer: RIKYU 1B ran a flat ~1.5 s per step whatever the batch (43k / 83k / 176k tokens/s at 8 / 16 / 32 seq per GPU). Skipping the plan when the tp mesh has size 1: 0.54 s, **242k** | RIKYU jobs 287003-287005 vs 287033-287035 | trainer skips TP at `--tp 1` |
| **Small models: FSDP2 alone wherever memory allows.** RIKYU 1B: 242k vs 71k (TP 4, 64 seq per group). Genoa 1B: 384-386 vs 263 (TP 4) | same | benchkit config |
| **Fugaku 1B needs TP 4.** FSDP2 alone fits 1 sequence per rank (6.1 GB, 166 tokens/s/node at 4 nodes) and runs out of memory at 2: the 128k-vocabulary logits and their gradients. TP 4 at 4 sequences per group: 175.5, 6.4 GB | Fugaku jobs 53648354-53648358 | benchkit config |
| **Both ends reach about a quarter of peak.** Model FLOPs (6N + attention, no recompute) against each machine's own peak: RIKYU ~25-26% of ~9 PF bf16 (4 x B200 dense), Fugaku ~22-28% of 6.1 TF fp32 (48 cores x 2 SVE pipes x 16 lanes x 2 x 2.0 GHz), Genoa ~25% of ~15 TF fp32. The ~1,400-1,700x per-node gap to RIKYU is the peak gap (~1,470x), mostly bf16 tensor cores vs fp32 SVE | fom lines above | - |
| **Batch on GPUs: the plateau, not the maximum.** RIKYU 30B per node: 1 seq 373, 8: 2,992, 32: 11,427, 64: 12,020, 128: 11,613-11,886 at ~2x memory | RIKYU jobs 286065-286079 | 64 in benchkit |
| **One agent per node under `srun` needs the node's CPUs.** benchkit's Slurm template asks for 4 tasks x `nthreads` CPUs; the inner `srun --ntasks=<nodes> --ntasks-per-node=1` that starts one torch.distributed.run agent per node inherits one task's `--cpus-per-task`, so the node's 4 workers shared 16 CPUs (64 with `--cpus-per-task=$((numproc_node * nthreads))`). Slurm warns the step exceeds the job's per-task CPUs, and runs. Throughput on RIKYU moved by at most 2% (GPU bound) | probe on RIKYU, jobs 288274-288280 | fixed in benchkit fsdp-llm (#343); omnilearned in #371 |
| **Launch with `venv/bin/python -m torch.distributed.run`, not `venv/bin/torchrun`.** The console script embeds the build directory's interpreter path; a relocated build fails or uses the old venv (benchkit review) | - | benchkit |

## Fugaku-specific notes from this work

- The 64-node 30B row ran 227, 238 and 271 s per step in three allocations with bit-identical
  losses: the spread is the allocation, not the image (the slowest ran on the CI-built SIF,
  the other two on an earlier SIF of the same image).
- benchkit's source adjustment (`_bk_source_path`) compared a resolved path with an
  unresolved one, so builds under `/vol0200` (a symlink to `/vol0002/mdt0`) failed with exit 2
  and no message. Fix in benchkit PR #371.
