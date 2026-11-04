# Head-to-head harness: container vs LLVM build

These are the scripts as run for [references/performance.md](../../references/performance.md) and
[references/overlap.md](../../references/overlap.md). They contain the tested site paths. Change
these before reuse:

| Path in the scripts | What it is |
|---|---|
| `/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross` | the cross-build workspace (`WORK`) |
| `.../pytorch-llvm-cross-largepage/rebuild-largepage` | the large-page rebuild: `bin/python3.11-fugaku`, `wheel-install/`, `runtime-site/` |
| `$WORK/pytorch-fugaku-b5879f2.sif` | the container image |
| `$WORK/h2h-paired/pf/pf-setup`, `pf-run` | copies of this repo's `fugaku/pf-setup` and `pf-run` |
| `$WORK/h2h-paired/site310`, `$WORK/h2h-tiny/site` | DeepSpeed 0.19.8 / transformers 4.57.6 for the container's python 3.10 and the LLVM build's 3.11 |
| `$WORK/h2h-profile/port`, `$WORK/h2h-async/backend*` | built `fjmpi_port` / `fjmpi_async` ([../backends](../backends)) |
| `$WORK/h2h-alloc/build/libmimalloc.so.2.2` | from [../alloc/build-mimalloc.sh](../alloc/build-mimalloc.sh) |
| `/vol0402/data/ra000009/u10035/miniconda3/bin/python3.11` | plain CPython 3.11, used only by the `plainpy*` presets |

The directory is expected at `$WORK/h2h-scale` (`rank.sh` binds `$WORK` into the container as `/cross`).

## Files

- `run.sh`: one job. It runs pf-setup, then each case in `H2H_CASES` in turn on the same allocation, each under `timeout $H2H_CASE_TIMEOUT`.
- `rank.sh`: one rank of one case. The cases are:
  - `container`: SIF with its own `fjmpi`
  - `llvm-port`: fjmpi_port
  - `llvm-sync`: fjmpi_async with `TORCH_FUGAKU_ASYNC=0`
  - `llvm-async`: queued allgathers
  - `container-sync` / `container-async`: the image build of fjmpi_async (`$WORK/h2h-async/container`)

  Suffixes modify a case:
  - `+PRESET` sets the allocator: `mi`, `mi-nohp`, `nohp`, `plainpy`, `*-af1/2`, `*-nommap`, `*-arena1/2`, `mi-*keep`
  - `@MB` sets `TORCH_FUGAKU_AG_STAGE_MB`
  - `%ALGO` sets `TORCH_FUGAKU_RS_ALGO` (`staged`, `a2a`, `ring`, `auto`) and uses the `h2h-rsalgo`
    builds of the backend: `$WORK/h2h-rsalgo/port` (LLVM) and `$WORK/h2h-rsalgo/container` (the
    image's torch_fugaku with a rebuilt `_c10d.so`, first on `PYTHONPATH`). See ../collectives.
- `run.py`: the driver (a wrapper around `llama_train.py`). It covers any node count and writes `timing.rank<N>.json` per rank: steps, phases with `getrusage` deltas, and fjmpi_async's blocked-time stats. Options:
  - `H2H_PROFILE_STEPS=a,b`: torch.profiler operator tables
  - `H2H_TRACE_STEPS=a,b`: fjmpi_async event trace
  - `H2H_PEAK=1`: per-phase peak RSS (perturbs timing)
  - `H2H_MEMSAMPLE=1`: a per-node memory sampler, set in `rank.sh` (perturbs timing badly)
- Layout and workload (`rank.sh`/`run.sh`): `H2H_RPN` ranks per node (threads = 48 / RPN; the container
  gets `PF_FJBLAS=1` only at 4), `H2H_SEQ` (default 128), `H2H_MB` micro-batch (default 1),
  `H2H_ZERO` stage (default 3; 0 for plain data parallel). Set `#PJM --mpi max-proc-per-node` to match.
  `H2H_DROPOUT` (gpt-* models), `H2H_THREADS`, `H2H_WAIT_POLICY` (diagnostics), and
  `H2H_FASTDROP=<libfastdrop.so>` (load the fast dropout kernel, LLVM only; see ../fastdrop).
- `llama_train.py`: Llama-architecture DeepSpeed ZeRO-3 trainer on synthetic tokens (models `llama-125m`, `1b`, ...).
- `summarize.py`: steady step time, losses and blocked time per case. Edit its job list.
- `compare_profiles.py`: self-CPU per operator, container vs LLVM, from two `profile.rank<N>.txt`.
- `analyze_trace.py`: per-gather slack, queueing, blocking and drains from one `trace.rank<N>.json`.

## Job

```sh
#PJM -L "rscgrp=small" -L "node=4" -L "elapse=00:30:00" -L "freq=2000,eco_state=0"
#PJM --mpi "max-proc-per-node=4"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM --llio cn-cache-size=384Mi
#PJM -g <group>
cd $WORK/h2h-scale
export H2H_CASES="container llvm-port+mi" H2H_MODEL=1b H2H_STEPS=14 H2H_WARMUP=6
bash ./run.sh
```

Run cases as **separate jobs** when they don't need to share an allocation. A failed case still
spends its timeout, and parallel jobs finish sooner. Compare only concurrent or same-allocation
cases: spread between allocations reached about 10%.

If you submit through an MCP job tool, set the launcher to `env`. Otherwise it may wrap `run.sh`
in `mpiexec -n N` and start every case N times.

Check per-rank exit codes (`<case>/rank*.exit.txt`), not PJM's state: `EXT` is not success.
