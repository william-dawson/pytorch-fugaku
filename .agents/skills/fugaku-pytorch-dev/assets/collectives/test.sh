#!/bin/bash
# Job: build the container backend, then tests/test_backend_mpi.py under each reduce-scatter
# algorithm (staged, a2a, ring, auto) in the image, 4 ranks per node.
set -uo pipefail
here=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-rsalgo
W=$(dirname "$here")
cd "$here"
out=$here/test.$PJM_JOBID; mkdir -p "$out"
bash build_container.sh > "$out/build.txt" 2>&1 || { echo "build failed" >> "$out/summary.txt"; exit 1; }
export PF_HOSTLIB="$here/.pf-hostlib.${PJM_JOBID}"
bash "$W/h2h-paired/pf/pf-setup" 4
set -a; source "$PF_HOSTLIB/pf.env"; set +a
for a in staged a2a ring auto; do
  TORCH_FUGAKU_RS_ALGO=$a OMP_NUM_THREADS=12 PF_BIND="$W:/cross:rw" \
    PYTHONPATH=/cross/h2h-rsalgo/container:/cross/h2h-paired/site310 \
    timeout 600 mpiexec -n "$WORLD_SIZE" -std-proc "$out/$a" bash "$W/h2h-paired/pf/pf-run" \
    "$W/pytorch-fugaku-b5879f2.sif" python -u /cross/h2h-rsalgo/test_backend_mpi.py
  echo "$a exit=$?" >> "$out/summary.txt"
done
