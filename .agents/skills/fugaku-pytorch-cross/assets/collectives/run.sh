#!/bin/bash
# Reduce-scatter variants, one mpiexec each (isolates each one's MPI memory), at the job's
# layout: RPN ranks per node, 48/RPN threads. Sizes: 4M and 50M floats per call (ZeRO-3's
# reduce bucket with --low-mem is 5e7). VARIANTS: a subset, or the allgathers (ag_*).
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
rpn=${RPN:?}; nodes=${PJM_NODE:?}
out="results.${PJM_JOBID}"; mkdir -p "$out"
export OMP_NUM_THREADS=$((48 / rpn)) OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=false
for v in ${VARIANTS:-a2a_staged a2a rsb ring halving allreduce}; do
  timeout 420 mpiexec -n $((nodes * rpn)) -std-proc "$out/$v" ./rs_bench $v 5 4194304 50000000
  echo "$v exit=$?" >> "$out/progress.txt"
done
