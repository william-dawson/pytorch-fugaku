# Experiment index (October-November 2026)

Every finding behind the pytorch-fugaku image's configuration, with where its evidence lives.
**Status** says what became of it:
- *image*: on by default in the SIF.
- *opt-in*: a flag or API in torch_fugaku or the examples.
- *LLVM*: the source-build route only.
- *dead end*: tried, did not help; don't repeat.
- *open*: unresolved.

All runs are on Fugaku A64FX, mostly benchkit Llama-1B (seq 2048, ZeRO-3) or the legacy 30B
shape. Only same-allocation comparisons count: spread between allocations reaches ~10%, and
2× for ring collectives on scattered (`:noncont`) allocations.

## Run time and launch

| Finding | Evidence | Status |
|---|---|---|
| LLIO's 128 MiB compute-node cache made big `.so` code refault: `--llio cn-cache-size=384Mi` | performance.md Fix 1 | image docs (job script) |
| torch's `ProcessGroupMPI` stalls 20-290 s per backward; use the container's `fjmpi` backend (or `fjmpi_port` for LLVM) | performance.md Fix 2 | image / LLVM |
| `OMP_WAIT_POLICY=PASSIVE`: 104.7 s/step | performance.md dead ends | dead end |
| mlock, `/dev/shm`, localtmp, /2ndfs, 4 GiB cache | performance.md dead ends | dead end |
| Samplers inside a rank's CPU set and `clear_refs` resets distort timing | performance.md pitfalls | method |

## Kernels and memory allocators

| Finding | Evidence | Status |
|---|---|---|
| Stock CPU `bernoulli_` was 71% of a GPT step; keyed-hash mask 50x faster | performance.md Fix 4 | image (torch_fugaku kernel) |
| LLVM build at 1B: libmpg refaults or OOMs; preload mimalloc | performance.md Fix 3, large-pages.md | LLVM |
| The container's torch has mimalloc built into its CPU allocator, whose arenas keep freed memory: 6.0 GB RSS per rank vs 0.46 GB of live tensors (30B, 8 layers); `MIMALLOC_PURGE_DELAY=0` → 1.23 GB between steps, peak 5.77 → 3.76 GB, at 50.9 → 54.6 s/step (+7%, one allocation, job 53596798) | thirty-b.md "Memory" | opt-in (memory-tight runs) |
| glibc `MALLOC_MMAP_THRESHOLD_` for that retention | thirty-b.md (job 53589763) | dead end |

## Layouts (ranks x threads per node)

| Finding | Evidence | Status |
|---|---|---|
| 4 × 12 computes 0.10 seq/s/node vs 0.078 (2 × 24) and 0.055 (1 × 48): ranks spanning CMGs lose locality | scaling.md | image docs |
| LLVM build ~40x slow below 4 ranks/node (every collective ~1.4 s); six fixes failed | comparisons.md | LLVM, open |
| Container scales 1B to 128 nodes at 4 × 12 (184 → 164, then 173 with fast ZeRO-3, tokens/s/node) | scaling.md | image docs |

## Collectives (`torch_fugaku/csrc/c10d_mpi.cpp`)

| Finding | Evidence | Status |
|---|---|---|
| Staged (8 MB) all-to-all reduce-scatter is latency bound from 64 ranks: 3.4-4.8 s vs 0.38-0.48 s unstaged at 128-256 ranks | performance.md Fix 5 | image (`auto`) |
| Ring reduce-scatter 4-5x faster at 4 ranks/node; reductions collapse below 4 ranks/node (`MPI_Allreduce` 46 s/call at 128 ranks, 2 × 24) | performance.md Fix 5 | image (`auto`) |
| What matters is the **job's** ranks per node: one-rank-per-node sub-groups in a 4-rank job (tensor parallel × FSDP) are fast in a ring | thirty-b.md "Communication" | image (`auto`) |
| Ring allgather in those groups: 49.67 → 42.09 s with ring reduce-scatter (30B, 8 layers, one allocation); in the group benchmark best at 32 nodes, worst at 64 | thirty-b.md, evidence/thirty-b-2026-11-04.json | image (`auto`), open (placement) |
| `MPI_Reduce_scatter_block` flat ~0.55 s per 50M floats at any scale; never the fastest | performance.md Fix 5 | - |
| Allgather does not collapse at any layout | performance.md Fix 5 | - |
| Non-contiguous allocations (`node=N`, the default up to 384 nodes) make ring timing vary up to 2× between allocations | evidence/thirty-b-2026-11-04.json | - |
| **Torus allocations flip the best algorithms.** 4 one-rank-per-node groups, 573 MB per call: on `4x6x4:torus` (96 nodes) and `4x6x6:torus` (144), `MPI_Allgather` takes 0.071-0.073 s (vs 0.37-0.54 non-contiguous) and the a2a reduce-scatter 0.43-0.44 s (vs 1.28), while rings get slower (0.72-0.77 s) (jobs 53596666, 53596667). Above 384 nodes, allocations are always torus | this page | open: training check, auto rule |
| `TORCH_FUGAKU_COMM_STATS=1|2`: per-kind collective time, with 2 the barrier skew apart | scaling.md | image |
| TCPStore rendezvous stops at 1012 clients: rank 0 hits the 1024 open-file soft limit (jobs 53593695 at 256 nodes, 53593699 at 576, both "1012/N clients joined"); `pf-run` raises it to the hard limit (262144) | this page | image (`fugaku/pf-run`) |
| With the limit raised, 12 of 1024 ranks still loop on "Connection reset by peer" and the store never completes (job 53597324; at 2304 workers torch warns `somaxconn` is 2048, job 53597327). `init_method="fjmpi://"` gives each process a local `HashStore` with rank/size from the launcher; fjmpi needs only MPI, and torch does not barrier on the store when creating groups (`TORCH_DIST_INIT_BARRIER=0`) | this page | image (`torch_fugaku.register_backend`) |
| Why `ulimit -n` in `pf-run` did not help (still 1012/N, jobs 53597324, 53597327): PJM caps open files per process with the `proc-openfd` resource (default 1024, max 1048576; `#PJM -L "proc-openfd=N"`); `somaxconn` (2048) is node-wide and not settable per job, so `env://` beyond it also needs staggered connects (per Fugaku support documentation, relayed in session) | this page | documented |
| First step at 128 nodes: forward 892 s vs 103 s at 64; at 576 nodes the second case in one allocation started in 328 s vs 1563 s. On the CI image alone at 256 nodes, two simultaneous allocations: first forward 70 s in one, 1195 s in the other (which ran `llio_transfer`, in 0 s, probably a no-op). So it depends on the allocation, not on the image or imports | thirty-b.md | open: find the late ranks |

## Overlap

| Finding | Evidence | Status |
|---|---|---|
| `MPI_Iallgather` 0% overlap; `MPIX_Allgather_init` (Tofu offload) ~98%, one offloaded request per process | overlap.md | - |
| Progress thread pays only with a core freed; compute-bound, so rejected | overlap.md | dead end |
| Thread-free progress from module hooks (`fjmpi_async`): +2.6% (LLVM 1.3B), ~12% of communication hidden at 64 nodes; the drain before other collectives cancels the gain | overlap.md | opt-in (LLVM harness), open |

## DeepSpeed ZeRO-3

| Finding | Evidence | Status |
|---|---|---|
| Per-rank Python loops in `reduce_scatter_coalesced` and `AllGatherCoalescedHandle.wait` grow linearly with ranks (0.8 → 4.3 s/step from 64 to 512); one strided copy per tensor: 50.31 → 47.44 s at 512 ranks, bit-identical | scaling.md "DeepSpeed ZeRO-3's per-rank loops" | image (`torch_fugaku.deepspeed`, automatic) |
| `model_cls(config)` builds the whole model on every rank (6.3 GB/rank at 1B); `_from_config` under `zero.Init`, then `_init_weights` on gathered parameters | scaling.md "Memory" | opt-in (`llama_train --zero-init`) |
| transformers' loss holds [2048, 128k] fp32 logits and gradients; 4 × 12 OOMed at random ranks at every node count; fused chunked loss | scaling.md "Memory" | opt-in (`torch_fugaku.nn`) |
| 30B under ZeRO-3: 4 × 12 and 2 × 24 out of memory, 1 × 48 fits but ~4x slower than legacy | thirty-b.md | dead end for 30B |

## 30B (legacy SIF shape)

| Finding | Evidence | Status |
|---|---|---|
| Legacy SIF: 576 nodes, 1 × 48, TP6 × PP8 × DP12, 870-923 s/iteration, ~6 tokens/s/node | comparisons.md, thirty-b.md | baseline |
| Tensor parallel 4 in the node × FSDP2 (`torch_fugaku.parallel`), sequence parallel, vocab-parallel loss: matches an unsharded copy to 1.6e-7 | thirty-b.md | opt-in (`examples/tp_train.py`) |
| Full 30B at 4 × 12: 6.0 / 5.7 / 7.6 / **6.6** tokens/s/node at 64 / 128 / 256 / 576 nodes (legacy SIF: ~6.0 at 576); compute 194.5 s/step everywhere, communication 28-46% | thirty-b.md "Full 48-layer runs" | measured |
| On a 576-node torus, a2a reduce-scatter + direct `MPI_Allgather` (the microbenchmark winners) were 470 s vs 312 s for rings in training | thirty-b.md | `auto` keeps rings |
| Ring reduce-scatter + direct `MPI_Allgather` at 256 nodes: 278.2 vs 221.9 s and 282.8 vs 263.9 s against rings, in two allocations | thirty-b.md (jobs 53608203, 53608204) | dead end |
| Full 30B on the CI image alone at 256 nodes: 9.23 tokens/s/node (221.9 s/step, 1.64 TFLOPs/node) | thirty-b.md (job 53608203) | measured |
| First step at 128 nodes: forward 892 s vs 103 s at 64 (one-off; later steps normal) | thirty-b.md | open |
| Communication 43% of the 64-node step; ~1.3 GB/s per node average, far below Tofu | thirty-b.md | open |

## Image verification

| Image | Check on Fugaku | Result |
|---|---|---|
| CI build d7fb755 (bundled DeepSpeed 0.19.8, transformers 4.57.6), image alone through the repository's `pf-setup`/`pf-run`, 2 nodes × 4 ranks (job 53607603) | `test_nn`; `test_backend_mpi` with `auto` and rings forced; `examples/tp_train.py --check --sp`; `examples/llama_train.py` ZeRO-3 over `fjmpi://` | all pass. 1.6e-7 against the unsharded copy; ZeRO-3 patch active without a flag; GPT-125M 4.58 s/step on 2 nodes |

| benchkit omnilearned (RIKEN-RCCS/benchkit `programs/omnilearned`), the October run directories rerun unchanged, normal mode, `BK_OMNILEARNED_SIF` the only difference, same allocation (jobs 53616203, 53616204) | events/s at 1 / 4 nodes: ed93ef4 17.0 / 55.3, **d7fb755 17.8 / 58.0** (README: 15.9 / 58.4); best losses 9.40150 / 9.56419 vs 9.40150 / 9.56420 | reproduces; current image ~5% faster |
| SIF reproducibility: `singularity pull` of the same OCI digest (`sha-ed93ef4`, `sha256:26d5947a…`) twice | same size (2,983,251,968 B), different SHA-256 (`ba657626…` in October, `47b102f0…` on 2026-11-05) | the OCI digest identifies an image; a SIF hash identifies one file |

Note: after `import deepspeed`, the import hooks have fired and removed themselves; check
`torch_fugaku.deepspeed._state`, not `sys.meta_path`.

## Where things are

- Harness: `assets/h2h/` (`run.sh`, `rank.sh` case syntax, `run.py`, trainers as run,
  `scale_summary.py`, `mem_probe.py`).
- Collective microbenchmark: `assets/collectives/rs_bench.cpp` (`SPLIT=k` for sub-groups).
- Evidence JSON: `references/evidence/`.
- The canonical trainers are the repository's `examples/`. The `assets/h2h/` copies are the
  versions that produced the recorded numbers.
