#!/bin/bash
# One rank of test_async.py on the LLVM largepage python; env:// rendezvous from pf.env (run.sh).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unset LD_PRELOAD PYTHONHOME LD_LIBRARY_PATH PYTHONPATH
export OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PROC_BIND=false OMP_WAIT_POLICY=ACTIVE PYTHONNOUSERSITE=1
rank=${PMIX_RANK:?}
export RANK=$rank LOCAL_RANK=$((rank % 4)) LOCAL_SIZE=4 USE_LIBUV=0 TORCH_FUGAKU_STAGE_MB=8
exec /vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross-largepage/rebuild-largepage/bin/python3.11-fugaku \
  -u "$here/test_async.py"
