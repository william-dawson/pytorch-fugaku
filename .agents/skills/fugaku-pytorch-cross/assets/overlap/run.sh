#!/bin/bash
# Overlap microbenchmark: every progress-thread mode x kernel, one mpiexec each so one
# failing mode does not hide the others.
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
out="results.${PJM_JOBID}"
mkdir -p "$out"
export OMP_NUM_THREADS=12 OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=false
sizes="262144 1048576 4194304 8388608"
for kernel in fma stream; do
  for mode in 0 3 1 2; do
    export OMPI_MCA_opal_progress_thread_mode=$mode
    log="$out/$kernel.mode$mode"
    timeout 300 mpiexec -n 8 -mca opal_progress_thread_mode $mode -std-proc "$log" \
      ./overlap $kernel 10 $sizes
    echo "$kernel mode$mode exit=$?" >> "$out/progress.txt"
  done
done
