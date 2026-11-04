# Container scaling: layouts, memory and communication (November 2026)

The container is the stack to scale. It runs every layout: 1 × 48, 2 × 24 and 4 × 12 (ranks ×
OpenMP threads per node). The LLVM build is limited to 4 × 12 (performance.md), and there it was
slower than the container at 32 and 128 nodes (146 vs 175 and 126 vs 164 tokens/s/node; jobs
53579580 / 53579593).

**Workload:** benchkit Llama-1B, seq 2048, micro-batch 1 per rank, ZeRO-3 `--low-mem`, fp32. The
global batch grows with the rank count (weak scaling).

**Setup:**
- `TORCH_FUGAKU_RS_ALGO=auto` (performance.md, Fix 5).
- `TORCH_FUGAKU_COMM_STATS=2`: every collective is timed by kind, after a barrier whose wait is
  reported as *skew*.
- `assets/h2h/scale_summary.py` reduces the per-rank timings. The data is in
  `evidence/container-scaling-2026-11-04.json`.
- 4 × 12 needs `--loss-chunk 256` (below).

## Results

Tokens/s per node:

| nodes | 1 × 48 | 2 × 24 | 4 × 12 |
|---:|---:|---:|---:|
| 4 | 91 | 135 | out of memory |
| 16 | 97 | 129 | **184** |
| 32 | 84 | 134 | **175** |
| 64 | 88 | 122 | **168** |
| 128 | 79 | 108 | **164** |

Communication share of the step (skew + allgather + reduce-scatter):

| nodes | 1 × 48 | 2 × 24 | 4 × 12 |
|---:|---:|---:|---:|
| 4 | 15% | 16% | |
| 16 | 18% | 20% | 12% |
| 32 | 23% | 16% | 15% |
| 64 | 26% | 21% | 16% |
| 128 | 29% | 27% | 14% |

Seconds per step, from 16 to 128 nodes (mean over ranks):

| layout | compute (everything outside collectives) | skew | allgather | reduce-scatter |
|---|---|---|---|---|
| 1 × 48 | 17.3 → 18.4 | 0.3 → 0.4 | 2.4 → 2.4 | **1.2 → 4.6** (staged, a2a from 64 ranks) |
| 2 × 24 | 25.5 → 27.7 | 0.4 → 0.6 | 3.5 → 4.1 | **2.4 → 5.5** (a2a from 64 ranks) |
| 4 × 12 | **39.0 → 42.8** | 1.9 → 3.0 (4.3 at 64) | 2.3 → 2.6 | 1.3 → 1.5 (ring) |

Every rank moves the same volume at every scale: 10.9 GB allgathered and 4.9 GB reduce-scattered
per step.

## What limits scaling

1. **Compute efficiency sets the order of the layouts.** Per node, compute finishes 0.10
   sequences/s at 4 × 12, 0.078 at 2 × 24 and 0.055 at 1 × 48. Ranks whose threads span several
   CMGs lose memory locality. At small scale the layouts differ in compute, not communication.
2. **At 1 × 48 and 2 × 24, reduce-scatter is the limit.** It grows about 4× from 16 to 128
   nodes for the same volume, while allgather stays flat. A ring would fix it, but reductions
   collapse below 4 ranks per node (performance.md, Fix 5).
   - The `auto` all-to-all already rescued 1 × 48 at 128 nodes: 25.8 s/step, against 595 s with
     the staged path (job 53502948).
3. **At 4 × 12, communication is not the limit.** It stays at 12-16%. What grows is compute
   (+3.8 s from 64 to 512 ranks) and skew. A profile at 16 nodes counts about 25,000 `aten::slice`
   and `aten::narrow` calls per step: per-rank chunking, likely DeepSpeed's. Whether that count
   grows with ranks is not yet measured.
4. **Memory is what stops 4 × 12** (next section).

## Memory

| layout | steady RSS per rank at 4 / 16 / 128 nodes | min free per node |
|---|---|---|
| 1 × 48 | 13.3 / 9.3 / 7.4 GB | 16-23 GB |
| 2 × 24 | 11.1 / 8.2 / 7.2 GB | 8-17 GB |
| 4 × 12 (chunked loss) | - / 6.2 / 5.2 GB | 4.2-5.3 GB |

- **ZeRO-3 shards shrink with nodes, then RSS plateaus.** At 16 B/param and 1.24 B parameters,
  the shard is 19.8 GB / ranks. The plateau of about 5-7 GB per rank does not shrink: runtime
  (~1.9 GB after `deepspeed.initialize`), activations, live gathered parameters and allocator
  retention.
- **The loss.** transformers' loss holds the [2048, 128256] float32 logits and their gradients.
  Without `--loss-chunk`, 4 × 12 failed at random ranks in step 1's forward at 4, 8, 24, 32, 64
  and 128 nodes (exit 137 or a bus error; jobs 53491470, 53495276, 53495290, 53504280, 53504294).
  `--loss-chunk 256` (`ChunkedLMLoss`) computes lm_head and cross-entropy in token chunks with the
  gradients in the forward, at the same FLOPs. It lowered steady RSS from 6.15 to 5.3 GB per rank,
  and 4 × 12 ran at 16, 32, 64 and 128 nodes. Losses match transformers' to float32 rounding
  (`test_chunked_loss.py`).
- **Model construction is the tightest moment.** VmHWM is 6.3 GB per rank before
  `deepspeed.initialize` shards the model, leaving 4.2 GB free on a 4 × 12 node. Building the
  model already partitioned would remove this.
- **Small node counts.** 4 × 12 at 4 nodes still runs out of memory (larger shards; job 53579562).
- **The margin is thin.** torch.profiler on every rank at 128 nodes tipped nodes over (job
  53581792). `run.py` now profiles rank 0 and the last rank only.

## Rule of thumb

Use 4 × 12 with `--loss-chunk` where it fits: 16+ nodes for 1B. Fall back to 2 × 24 where memory
is short (small node counts, bigger models). 1 × 48 has the most headroom but is about half as
fast.
