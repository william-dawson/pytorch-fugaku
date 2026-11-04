#!/bin/bash
# fjmpi_async correctness + overlap: queued (ASYNC=1), then synchronous (ASYNC=0) on the same binary.
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
out="results.${PJM_JOBID}"
mkdir -p "$out"
export PF_HOSTLIB="$here/.pf-hostlib.${PJM_JOBID}"
bash "$here/../../h2h-paired/pf/pf-setup" 4
set -a; source "$PF_HOSTLIB/pf.env"; set +a
for a in 1 0; do
  export TORCH_FUGAKU_ASYNC=$a
  timeout 600 mpiexec -n "$WORLD_SIZE" -std-proc "$out/async$a" /bin/bash "$here/rank.sh"
  echo "async$a exit=$?" >> "$out/progress.txt"
done
