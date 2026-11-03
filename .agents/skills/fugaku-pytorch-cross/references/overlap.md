# Overlapping communication with compute (work in progress)

With the fixes in [performance.md](performance.md), a large share of each step is spent blocked in
collectives. Measured with `fjmpi_async`'s sync mode, which records blocked time per step:

| | Step | Allgather | Reduce-scatter |
|---|---:|---:|---:|
| 125M, 8 nodes | 2.69 s | 0.29 s | 0.52 s |
| 1B, 4 nodes | 12.5 s | 3.6 s | 2.8 s |

The 1B row is from before the mimalloc fix. Both fjmpi and fjmpi_port are fully synchronous, so
none of this overlaps with compute. This page records what Fujitsu MPI can overlap, what was
built, and why it does not help training yet.

## What Fujitsu MPI can overlap (microbenchmark)

`assets/overlap/overlap.cpp` (Fujitsu compiler, libfjomp + libmpg like the torch build) times an
allgather alone, a 12-thread OpenMP kernel sized to match it, and the two together five ways.
Every mode of `opal_progress_thread_mode` was tested. Overlap is
`(comm + compute - together) / min(comm, compute)`. Job 53462090: 2 nodes x 4 ranks x 12 threads;
all 160 cases passed the data check.

| Bytes per rank | Comm alone | Iallgather + Wait | + 16 MPI_Test | Helper thread, blocking | Persistent `MPIX_Allgather_init` |
|---|---:|---:|---:|---:|---:|
| 256K | 0.76 ms | -6% | 79% | 12% | **94%** |
| 1M | 2.99 ms | 5% | 90% | 34% | **97%** |
| 4M | 12.1 ms | 3% | 88% | 39% | **98%** |
| 8M | 23.7 ms | 2% | 89% | -11% | **98%** |

Default mode, compute-bound kernel; a memory-bound kernel gives the same picture.

- **Plain non-blocking collectives get no overlap.** The data moves when `MPI_Wait` is called. Fujitsu's MPI guide warns about this (§6.2).
- **The assistant-core progress thread does not help.** `opal_progress_thread_mode` 1 and 2, with `FJMPI_Progress_start/stop` around the compute, gave 0%. **Mode 3 is harmful:** from 4 MB per rank it adds about 200 ms to every operation. Never use mode 3.
- **Persistent allgather overlaps.** `MPIX_Allgather_init` is offloaded to the Tofu interconnect from its second start (guide §5.5.3). It hides 94-98% with no polling, and its transfer is about 18% faster (19.4 vs 23.7 ms at 8 MB). Limits: only one such offloaded request per process, basic datatypes, an intra-communicator and a size condition.
- **Polling also works.** Calling `MPI_Test` 16 times during the compute gave 85-95%.
- **torch's ProcessGroupMPI design is weak.** A helper thread doing a blocking allgather gave 12-40% overlap, and was slower than serial at 8 MB.

## `fjmpi_async` (assets/backends/fjmpi_async)

This is fjmpi_port plus queued allgathers. `_allgather_base` and `allgather_into_tensor_coalesced`
return a pending `Work`. One persistent `MPIX_Allgather_init` request on a dedicated communicator
moves the queue through a fixed buffer (`TORCH_FUGAKU_AG_STAGE_MB` for all ranks, default 64).
The next transfer starts whenever the backend is entered.

Every other collective first **drains** the queue. Without that, a rank blocked in a collective on
the main communicator could hold back a gather transfer another rank waits for.

`TORCH_FUGAKU_ASYNC=0` gives the synchronous path on the same binary. `stats()` reports seconds
blocked by kind; with `TORCH_FUGAKU_TRACE=1`, `trace()` returns timestamped events for
`assets/h2h/analyze_trace.py`.

**Correctness** (`test/`, job 53463810):
- 27 queued gathers of mixed sizes and dtypes, with an `all_reduce` in the middle and waits in reverse order: zero mismatches on all 8 ranks, in both modes.
- Completion by polling `is_completed()` alone: passed.
- Torch-level overlap (8 MB per rank against matmuls): 91% hidden, 23.3 ms overlapped vs 31.7 ms serial.

**Training: no speedup.** Results from `h2h-scale` (`llvm-sync` vs `llvm-async`):

| Case | Sync | Async (buffer size) | Job |
|---|---:|---:|---|
| 125M, 8 nodes | 2.69 s | 4.16 s (64 MB) | 53464099 |
| 125M, 8 nodes | 2.60 s | 2.86 s (8 or 16 MB) | 53465717 |
| 1B, 4 nodes, large pages off | 12.51 s | 12.66 s (16 MB) | 53467996 |

Losses are identical in every case.

Event traces (`H2H_TRACE_STEPS`, jobs 53468017 and 53467996) explain why:

| Two traced steps, rank 0 | 125M forward | 1B forward | 1B backward |
|---|---:|---:|---:|
| Slack: issue to wait, median | 507 ms | 242 ms | 447 ms |
| Queued before first transfer, median | 495 ms | 214 ms | 49 ms |
| Backend entered while moving | 0 times for 40/50 | 0 for 40/126 | 0 for 50/96 |
| Finished before the caller waited | 2/50 | 0/126 | 90/96 |

1. **DeepSpeed gives plenty of slack.** ZeRO-3 prefetch issues gathers 0.25-1.3 s before it waits on them.
2. **Transfers sit idle.** A transfer starts or advances only when the backend is entered, and nothing enters it during compute. Gathers wait in the queue and move at `wait()`, which is synchronous plus overhead.
3. **Padding.** Every transfer is full size, because the persistent count is fixed. The optimizer step's 33 near-empty gathers each cost a whole transfer: that phase went from 0.07 to 0.26 s at 125M. With a 64 MB buffer, 86 transfers moved 5.5 GB for 0.9 GB of data.
4. **Draining.** At 1B, 90 of 96 backward gathers did finish early. But every reduce-scatter drains the queue first, which cost 3.3 s over two steps.

## Packing: do not use `backend-pack`

A variant packed every queued gather into each transfer. Its contents depended on what one rank
happened to have queued when the transfer started. Timing differs across ranks, so ranks can pack
different gathers into the same transfer and silently exchange the wrong data. Its 2-node test
passed only because the test queued faster than any transfer completed. It is not in the skill and
was never used for training. **Rule: transfer boundaries must depend only on what every rank sees
identically (call order and sizes), never on timing.**

## Next steps, in order of expected payoff

1. **A progress engine.** A thread that sleeps about 100 µs between `MPI_Test`s on the transfer in flight, copies it out and starts the next, so the queue drains during compute. Unlike the helper-thread result above, it would not sit inside a blocking call.
2. **Wait for reduce-scatter while still pumping gathers,** using the non-blocking call plus a test loop, instead of draining. Worth about 1.6 s per step at 1B.
3. **Route small gathers by size** to an exact-size `MPI_Iallgather` on their own communicator, issued immediately. Size is identical on every rank, so this is safe where packing was not.
4. Then reduce-scatter itself: there is no documented Tofu offload, so poll.
