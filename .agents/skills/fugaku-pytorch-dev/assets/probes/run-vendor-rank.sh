#!/bin/bash
set -euo pipefail
ranks=${1:?}; threads=${2:?}
rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
unset LD_PRELOAD
export LD_LIBRARY_PATH=/opt/FJSVxtclanga/tcsds-1.2.43/lib64
export OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND=false OMP_WAIT_POLICY=ACTIVE
exec ./vendor-reference "$threads" "metrics.vendor.r$ranks.${PJM_JOBID}.csv" \
  > "rank.vendor.r$ranks.${PJM_JOBID}.$rank.log" 2>&1
