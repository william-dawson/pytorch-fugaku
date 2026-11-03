#!/bin/bash
# One rank, one CMG (12 threads), on the LLVM largepage python.
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unset LD_PRELOAD PYTHONHOME LD_LIBRARY_PATH PYTHONPATH
export OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PROC_BIND=false OMP_WAIT_POLICY=ACTIVE PYTHONNOUSERSITE=1
/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross-largepage/rebuild-largepage/bin/python3.11-fugaku -u \
  "$here/test_fastdrop.py" > "$here/test.$PJM_JOBID.txt" 2>&1
echo "exit=$?" >> "$here/test.$PJM_JOBID.txt"
