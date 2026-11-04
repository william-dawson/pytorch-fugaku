# 30B on the container at 4 × 12: tensor parallel × FSDP2 (November 2026)

**Target:** the legacy SIF's only real-scale run, `30b_pp8_tp6_dp12_*.sh` (Megatron-DeepSpeed on
Fujitsu's PyTorch 1.13, December 2023):
- GPT-30B: 48 layers, hidden 6912, 72 heads, seq 2048, global batch 1536.
- 576 nodes as an 8×12×6 torus, **1 rank × 48 threads per node**.
- Tensor parallel 6 (across 6 nodes over Tofu) × pipeline 8 × data parallel 12; 128 micro-batches
  per pipeline per step.
- Measured (jobs 29716987-29757867): 870-923 s per iteration, 1.38-1.46 TFLOPs per node
  (Megatron's count), about 6 tokens/s per node.

The new SIF's container runs it at 4 × 12 with PyTorch-native 2-D parallelism
(`assets/h2h/tp_train.py`):
- **Tensor parallel 4 inside each node.** The 4 ranks of a node are one tensor-parallel group.
  - q/k/v and gate/up are column-parallel; o and down are row-parallel.
  - The embedding, `lm_head` and the loss are vocab-parallel (`loss_parallel`), so no rank holds
    the [vocab, hidden] weight or the [tokens, vocab] logits.
- **Sequence parallel (`--sp`).** The residual stream, the norms and the checkpointed layer
  inputs are sharded along the sequence within the node.
- **FSDP2 (`fully_shard`) across nodes**, per layer plus the embedding and `lm_head`, with
  activation checkpointing per layer.
- **Construction.** The model is built on the meta device and materialized as shards.
- **Model.** A Llama with the legacy shape: full multi-head attention and a SwiGLU MLP of 2/3 × 4h
  give each layer GPT's 12h² weights, 573M per layer. Vocabulary 50304, untied embeddings, no
  dropout (the legacy run had GPT-2's dropout).

**Correctness.** `--check` trains an unsharded copy on the same global batch on every rank. Tiny
Llama, 2 nodes, TP 4 × DP 2, 4 AdamW steps: worst relative loss error 1.6e-7, with and without
`--sp` (jobs 53586749, 53590278).

## Why not ZeRO-3 (DeepSpeed, data parallel only)

At 30B width, under ZeRO-3 every rank needs these at once:
- the gathered vocabulary embedding (348M parameters, 1.39 GB);
- its full gradient (1.39 GB);
- DeepSpeed's reduce-scatter copy of that gradient (1.39 GB);
- the loss's gradient buffer.

None of it shrinks with more nodes. `llama_train.py --model gpt-30b --layers N --zero-init
--fast-zero --loss-chunk 256`, 32 nodes:

| layout | 4 layers | 8 layers | per-rank peak |
|---|---|---|---|
| 4 × 12 | out of memory, step 0 backward (53584326) | timed out at 1.4 GB free (53584338) | - |
| 2 × 24 | out of memory (53587234) | out of memory (53587246) | 13.5-13.7 GB |
| 1 × 48 | 109 s/step (53587253) | 211 s/step (53587257) | 15.4 / 15.8 GB |

- **2 × 24 and 4 × 12 cannot fit.** Per-rank memory is about 15 GB with only ~0.1 GB per
  layer, so it is the fixed part that doesn't fit, not the layers.
- **1 × 48 fits from about 128 nodes, but it is slow.** At ~25.5 s per layer, the full model
  would run at about 1.7 tokens/s per node, roughly 4× slower than the legacy run.

## Memory: torch's mimalloc keeps freed memory

In the container's torch 2.11 (stock aarch64 wheel), the CPU allocator uses a bundled mimalloc.

- **Symptom.** At 32 nodes, the tensor-parallel 4-layer probe peaked at 4.94 GB per rank and the
  8-layer one at 6.11 GB (53587159 / 53587170). That projected to about 17 GB at 48 layers.
- **Ruled out:**
  - **Sharded state.** At 64 nodes the 8-layer peak was still 6.05 GB (53589768).
  - **glibc retention.** `MALLOC_MMAP_THRESHOLD_` changed nothing (53589763).
- **The live-tensor census disagrees with RSS.** `tp_train.py --census` sums the storage of
  parameters, gradients, optimizer state and every tensor the garbage collector sees. Between
  steps, 8 layers hold 0.15 GB of parameters and 0.31 GB of Adam state, yet VmRSS is 6.0 GB
  (53590349).
- **smaps shows where it is.** It sits in 9 anonymous 1-2 GB mappings (12.3 GB reserved,
  4.2-5.3 GB resident): mimalloc arenas, not tensors.
- **Fix.** `MIMALLOC_PURGE_DELAY=0` (mimalloc reads its environment even when bundled):

| 30B, 8 layers, 32 nodes | between steps | peak | step |
|---|---:|---:|---:|
| default | 6.0 GB | 6.13 GB | 42.4 s (53590349) |
| `MIMALLOC_PURGE_DELAY=0` | **1.23 GB** | **4.12 GB** | 48.4 s (53590898, other allocation) |

Within one allocation (job 53596798, `--sp`): 50.9 → 54.6 s per step (+7%), peak 5.77 → 3.76 GB,
identical losses. So it is opt-in, for memory-tight runs, not an image default.

## Communication: FSDP's groups have one rank per node

With tensor parallel 4 inside the node, each FSDP group holds the same tensor-parallel slot on
every node: one rank per node, and 4 such groups per node communicate at once.

`rs_bench` with `SPLIT=4` reproduces that shape. 32 nodes; seconds per 573 MB call, one layer's
tensor-parallel slice (53589753):

| reduce-scatter | ring **0.38** | a2a 0.72 | staged 0.82 | rsb 1.67 |
|---|---|---|---|---|
| **allgather** | ring **0.35** | persistent 0.35 | `MPI_Allgather` 0.42 | staged 0.43 |

Hence the backend's `auto` rules (performance.md, Fix 5): rings when the job runs 4+ ranks per
node. Same allocation, 30B with 8 layers, 32 nodes, identical losses (53591331):

| `RS_ALGO,AG_ALGO` | step | TFLOPs/node |
|---|---:|---:|
| staged,staged (old `auto`) | 49.67 s | 1.36 |
| ring,staged | 44.65 s | 1.52 |
| ring,ring (new `auto`) | **42.09 s** | **1.61** |

## Run it

```sh
# h2h harness (assets/h2h), container case, 4 ranks x 12 threads per node
export H2H_CASES='container%auto' H2H_TRAINER=tp_train.py H2H_TP=4 H2H_TP_SP=1 \
       H2H_MODEL=30b H2H_SEQ=2048 H2H_RPN=4 H2H_STEPS=4 H2H_WARMUP=1 MIMALLOC_PURGE_DELAY=0
bash ./run.sh        # #PJM --mpi max-proc-per-node=4, --llio cn-cache-size=384Mi
```

`mem_probe.py results.<job>/<case>` reports the max-rank memory per phase and the lowest free
memory per node (with `H2H_MEMSAMPLE=1`).

## Open

- The full 48-layer run (64 and 128 nodes) is in progress; numbers to follow.
- `--sp`'s time cost against plain tensor parallel.
- The global batch is one sequence per tensor-parallel group per step (64 at 256 nodes). Matching
  the legacy 1536 needs gradient accumulation, or more nodes.
