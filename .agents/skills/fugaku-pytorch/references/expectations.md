# What to expect (measured, November 2026)

**Conditions:** normal mode (`freq=2000,eco_state=0`), fp32, seq 2048, one sequence per
data-parallel rank per step. Steady steps only. Same-allocation comparisons only: allocations
differ by up to ~10%.

Sources and job IDs are in the `fugaku-pytorch-dev` skill: `references/scaling.md`,
`references/thirty-b.md`, `references/experiments.md`.

## Llama-1B, DeepSpeed ZeRO-3 (benchkit configuration)

Tokens/s per node:

| nodes | 1 × 48 | 2 × 24 | 4 × 12 (chunked loss) |
|---:|---:|---:|---:|
| 4 | 91 | 135 | out of memory |
| 16 | 97 | 129 | 184 |
| 32 | 84 | 134 | 175 |
| 64 | 88 | 122 | 168 |
| 128 | 79 | 108 | 164 (173 with the ZeRO-3 loop patch, now default) |

Communication share at 128 nodes:
- 1 × 48: 29%;
- 2 × 24: 27%, mostly reduce-scatter;
- 4 × 12: 14%.

Per-rank steady RSS at 4 × 12 is about 5.2-6.2 GB.

## 30B (Llama with the legacy GPT-30B shape), tensor parallel 4 × FSDP2 at 4 × 12, `--sp`

| nodes | step | tokens/s per node | peak per rank | communication |
|---:|---:|---:|---:|---:|
| 64 | 341 s | 6.0 | 5.1 GB | 43% |
| 128 | 358 s | 5.7 | 4.5 GB | 46% |
| 256 | 269 s | 7.6 | 4.2 GB | 28% |
| 576 | 312 s | **6.6** | 4.0 GB | 38% |

**Legacy Fugaku-LLM Megatron SIF**, GPT-30B on 576 nodes (1 × 48, TP6 × PP8 × DP12): ~6 tokens/s
per node.

- Compute is 194.5 s per step at every node count; communication depends on the allocation.
- At 576 nodes this is about 10% more per node than the legacy SIF on the same node count.
- The first step of a cold start takes 3-5x a steady step, in its first forward pass. A second
  launch in the same allocation started in 328 s instead of 1563 s, so it is cache warming.
- Use `init_method="fjmpi://"`. 256 and 576 nodes only started with it.
- ZeRO-3 cannot run 30B at 4 × 12 or 2 × 24 (out of memory). 1 × 48 fits, but runs about 1.7
  tokens/s per node.

## Memory per node

About 28 GB is usable out of 32 GB of HBM.

| layout | per rank |
|---|---|
| 4 × 12 | ~7 GB |
| 2 × 24 | ~14 GB |
| 1 × 48 | ~28 GB |

At 4 × 12, the vocabulary loss and the model construction were the peaks before
`chunked_lm_loss` and `zero.Init` construction.
