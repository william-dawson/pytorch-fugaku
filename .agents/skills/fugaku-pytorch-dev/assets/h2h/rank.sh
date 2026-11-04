#!/bin/bash
# One rank of one case. Cases:
#   container    pytorch-fugaku SIF, its own fjmpi backend
#   llvm-port    LLVM largepage build, fjmpi_port (synchronous; the fix of h2h-mlock-test)
#   llvm-sync    LLVM, fjmpi_async with TORCH_FUGAKU_ASYNC=0 (same as port, plus blocked-time stats)
#   llvm-async   LLVM, fjmpi_async with queued, Tofu-offloaded allgathers
# Model and step counts come from H2H_MODEL, H2H_STEPS, H2H_WARMUP (run.sh).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(dirname -- "$here")
case=${1:?expected container, llvm-port, llvm-sync or llvm-async}
# CASE^: that case with llama_train --fast-zero (ZeRO-3 bookkeeping without per-rank loops).
if [[ "$case" == *^ ]]; then
  export H2H_FAST_ZERO=1
  case=${case%^}
fi
# CASE%RS[,AG]: that case with TORCH_FUGAKU_RS_ALGO=RS (staged, a2a, ring, auto), optionally
# TORCH_FUGAKU_AG_ALGO=AG (staged, ring, auto), and the h2h-rsalgo builds of the backend
# (container: its torch_fugaku first on PYTHONPATH; llvm: fjmpi_port).
if [[ "$case" == *%* ]]; then
  algo=${case##*%}
  export TORCH_FUGAKU_RS_ALGO=${algo%%,*} H2H_PORT_PATH=$root/h2h-rsalgo/port
  [[ "$algo" == *,* ]] && export TORCH_FUGAKU_AG_ALGO=${algo#*,}
  rs_path=/cross/h2h-rsalgo/${H2H_OVERLAY:-container}:
  case=${case%\%*}
fi
# llvm-*+PRESET: libmpg (the LLVM python's malloc) settings; at 1B the LLVM build took ~30x the
# container's page faults with large pages off (jobs 53465728 vs 53463764).
if [[ "$case" == *+* ]]; then
  preset=${case#*+}
  case=${case%+*}
  case "$preset" in
    nohp) export XOS_MMM_L_HPAGE_TYPE=none ;;
    nohp-af1) export XOS_MMM_L_HPAGE_TYPE=none XOS_MMM_L_ARENA_FREE=1 ;;
    nohp-af2) export XOS_MMM_L_HPAGE_TYPE=none XOS_MMM_L_ARENA_FREE=2 ;;
    hp-af1) export XOS_MMM_L_ARENA_FREE=1 ;;
    hp-af2) export XOS_MMM_L_ARENA_FREE=2 ;;
    # torch allocates with posix_memalign (glibc; libmpg overrides only malloc/free/realloc).
    # nommap: glibc never mmaps an allocation, so freed tensors stay in the heap for reuse.
    nohp-nommap) export XOS_MMM_L_HPAGE_TYPE=none MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736 ;;
    # plainpy: miniconda's python3.11, which does not link libmpg: glibc malloc throughout.
    plainpy) export H2H_PYTHON=/vol0402/data/ra000009/u10035/miniconda3/bin/python3.11 ;;
    # arenaN: at most N glibc malloc arenas (default 8 x cores), so memory freed by one thread
    # is reusable by another instead of fragmenting per-thread heaps.
    plainpy-arena1) export H2H_PYTHON=/vol0402/data/ra000009/u10035/miniconda3/bin/python3.11 MALLOC_ARENA_MAX=1 ;;
    plainpy-arena2) export H2H_PYTHON=/vol0402/data/ra000009/u10035/miniconda3/bin/python3.11 MALLOC_ARENA_MAX=2 ;;
    # mi*: mimalloc preloaded into the python process only (h2h-alloc); a preloaded malloc wins
    # over libmpg's, and covers posix_memalign (torch's tensors). keep: never return memory.
    # mipurge0: torch's bundled mimalloc (container) returns freed memory at once.
    mipurge0) export MIMALLOC_PURGE_DELAY=0 ;;
    mi) export H2H_LD_PRELOAD=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-alloc/build/libmimalloc.so.2.2 ;;
    # mi-nodecommit: mimalloc purges by resetting pages instead of decommitting them.
    mi-nodecommit) export H2H_LD_PRELOAD=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-alloc/build/libmimalloc.so.2.2 MIMALLOC_PURGE_DECOMMITS=0 ;;
    mi-nohp) export XOS_MMM_L_HPAGE_TYPE=none H2H_LD_PRELOAD=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-alloc/build/libmimalloc.so.2.2 ;;
    mi-nohp-keep) export XOS_MMM_L_HPAGE_TYPE=none H2H_LD_PRELOAD=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-alloc/build/libmimalloc.so.2.2 MIMALLOC_PURGE_DELAY=-1 ;;
    mi-keep) export H2H_LD_PRELOAD=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-alloc/build/libmimalloc.so.2.2 MIMALLOC_PURGE_DELAY=-1 ;;
    plainpy-nommap) export H2H_PYTHON=/vol0402/data/ra000009/u10035/miniconda3/bin/python3.11 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736 ;;
    *) echo "unknown preset $preset" >&2; exit 2 ;;
  esac
fi
# llvm-async@MB: that case with TORCH_FUGAKU_AG_STAGE_MB=MB (the gather buffer for all ranks).
if [[ "$case" == *@* ]]; then
  export TORCH_FUGAKU_AG_STAGE_MB=${case#*@}
  case=${case%@*}
fi
unset LD_PRELOAD PYTHONHOME LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 DS_ACCELERATOR=cpu
rpn=${H2H_RPN:-4}
# H2H_THREADS: fewer OpenMP threads than the rank's cores (e.g. to leave cores for MPI helpers).
export OMP_NUM_THREADS=${H2H_THREADS:-$((48 / rpn))} OMP_DYNAMIC=FALSE OMP_PROC_BIND=false OMP_WAIT_POLICY=${H2H_WAIT_POLICY:-ACTIVE} OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MEMLOG=1 NNODES=$(( ${PJM_NODE//x/*} ))
rank=${PMIX_RANK:?}
out=${H2H_OUTPUT:?}
# H2H_MEMSAMPLE=1: one sampler per node (its first rank): node MemAvailable and every training
# process's RSS, every 2 s, so a case killed for memory still shows who grew and when. Opt-in:
# it runs inside that rank's CPU set and its preemptions stalled every collective (job 53475550).
sampler=
if [[ "${H2H_MEMSAMPLE:-0}" == 1 ]] && (( rank % rpn == 0 )); then
  (while :; do
     printf '%s avail_kb=%s' "$(date +%s.%N)" "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
     ps -u "$(id -u)" -o rss=,comm= | awk '$2 ~ /python/ {printf " %s", $1} END {print ""}'
     sleep 2
   done) > "$out/mem.node$((rank / rpn)).txt" 2>/dev/null &
  sampler=$!
fi
trap 'code=$?; [[ -n "$sampler" ]] && kill "$sampler" 2>/dev/null; printf "%s\n" "$code" > "$out/rank'"$rank"'.exit.txt"' EXIT
args=(--model "${H2H_MODEL:?}" --seq "${H2H_SEQ:-128}" --micro-batch "${H2H_MB:-1}" --dtype fp32 --zero "${H2H_ZERO:-3}" --steps "${H2H_STEPS:?}"
  --warmup "${H2H_WARMUP:?}" --attn sdpa_causal --checkpointing --low-mem)
# gpt-* models only: llama_train's --dropout (transformers' default 0.1, as Megatron).
[[ -n "${H2H_DROPOUT:-}" ]] && args+=(--dropout "$H2H_DROPOUT")
# H2H_LOSS_CHUNK=N: llama_train's fused chunked lm_head + loss (no [tokens, vocab] logits).
[[ -n "${H2H_LOSS_CHUNK:-}" ]] && args+=(--loss-chunk "$H2H_LOSS_CHUNK")
# H2H_LAYERS=N: override the layer count (memory probes); H2H_ZERO_INIT=1: build the model partitioned.
[[ -n "${H2H_LAYERS:-}" ]] && args+=(--layers "$H2H_LAYERS")
[[ "${H2H_ZERO_INIT:-0}" == 1 ]] && args+=(--zero-init)
[[ "${H2H_FAST_ZERO:-0}" == 1 ]] && args+=(--fast-zero)
# H2H_TRAINER=tp_train.py (run.py): tensor parallel x FSDP2; H2H_TP ranks per tensor-parallel group,
# H2H_TP_CHECK=1 compares against an unsharded copy (tiny models). llama_train's own flags go too.
[[ -n "${H2H_TP:-}" ]] && args+=(--tp "$H2H_TP")
[[ "${H2H_TP_CHECK:-0}" == 1 ]] && args+=(--check)
[[ "${H2H_TP_SP:-0}" == 1 ]] && args+=(--sp)
[[ "${H2H_TP_CENSUS:-0}" == 1 ]] && args+=(--census)
if [[ "$case" == container* ]]; then
  # container-sync / container-async: the image build of fjmpi_async (h2h-async/container),
  # synchronous or with queued, Tofu-offloaded allgathers; plain "container" is the image's fjmpi.
  cbackend=fjmpi
  case "$case" in
    container-sync) cbackend=fjmpi_async; export TORCH_FUGAKU_ASYNC=0 ;;
    container-async) cbackend=fjmpi_async; export TORCH_FUGAKU_ASYNC=1 ;;
  esac
  [[ "$cbackend" == fjmpi_async ]] && export H2H_CONTAINER_BACKEND=fjmpi_async H2H_ASYNC_PATH=/cross/h2h-async/container
  # Fujitsu BLAS only at one rank per CMG, as benchkit PR 343 (wider ranks: README deadlock warning).
  export H2H_VARIANT=container PF_BIND="$root:/cross:rw" PYTHONPATH=${rs_path:-}/cross/h2h-paired/site310
  if (( rpn == 4 )); then export PF_FJBLAS=1; fi
  export H2H_OUTPUT="/cross/h2h-scale/${H2H_OUTPUT_REL:?}"
  bash "$root/h2h-paired/pf/pf-run" "$root/pytorch-fugaku-b5879f2.sif" python -u /cross/h2h-scale/run.py \
    "${args[@]}" --backend "$cbackend" > "$out/rank$rank.log" 2>&1
  exit
fi
export H2H_VARIANT=llvm RANK=$rank LOCAL_RANK=$((rank % rpn)) LOCAL_SIZE=$rpn USE_LIBUV=0 TORCH_FUGAKU_STAGE_MB=${TORCH_FUGAKU_STAGE_MB:-8}
case "$case" in
  llvm-port) export H2H_BACKEND=fjmpi_port ;;
  llvm-sync) export H2H_BACKEND=fjmpi_async TORCH_FUGAKU_ASYNC=0 ;;
  llvm-async) export H2H_BACKEND=fjmpi_async TORCH_FUGAKU_ASYNC=1 ;;
  *) echo "unknown case $case" >&2; exit 2 ;;
esac
env ${H2H_LD_PRELOAD:+LD_PRELOAD=$H2H_LD_PRELOAD} "${H2H_PYTHON:-/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross-largepage/rebuild-largepage/bin/python3.11-fugaku}" -u \
  "$here/run.py" "${args[@]}" --backend "$H2H_BACKEND" > "$out/rank$rank.log" 2>&1
