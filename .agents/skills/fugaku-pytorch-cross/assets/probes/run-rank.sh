#!/bin/bash
set -euo pipefail
compiler=${1:?}; ranks=${2:?}; threads=${3:?}
case "$compiler" in native|llvm) ;; *) exit 2 ;; esac
rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-}}
[[ -n "$rank" ]] || exit 99
fj=/opt/FJSVxtclanga/tcsds-1.2.43/lib64
export LD_LIBRARY_PATH="$fj"
export LD_PRELOAD="$fj/libfjomphk.so:$fj/libfjomp.so"
export OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=false OMP_WAIT_POLICY=ACTIVE
exec "./bench-$compiler" "$threads" "metrics.$compiler.r$ranks.${PJM_JOBID}.csv" \
  > "rank.$compiler.r$ranks.${PJM_JOBID}.$rank.log" 2>&1
