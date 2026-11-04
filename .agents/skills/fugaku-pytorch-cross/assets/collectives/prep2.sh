#!/bin/bash
# 2 nodes: rebuild the container backend, test_backend_mpi.py with ring allgather and ring
# reduce-scatter forced (and auto), then tp_train's --sp check through the h2h harness.
set -uo pipefail
here=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-rsalgo
W=$(dirname "$here")
cd "$here"
out=$here/prep2.$PJM_JOBID; mkdir -p "$out"
bash build_container.sh > "$out/build.txt" 2>&1; echo "build exit=$?" >> "$out/summary.txt"
export PF_HOSTLIB="$here/.pf-hostlib.${PJM_JOBID}"
bash "$W/h2h-paired/pf/pf-setup" 4
set -a; source "$PF_HOSTLIB/pf.env"; set +a
for algos in "ring ring" "auto auto"; do set -- $algos
  TORCH_FUGAKU_AG_ALGO=$1 TORCH_FUGAKU_RS_ALGO=$2 OMP_NUM_THREADS=12 PF_BIND="$W:/cross:rw" \
    PYTHONPATH=/cross/h2h-rsalgo/container:/cross/h2h-paired/site310 \
    timeout 600 mpiexec -n "$WORLD_SIZE" -std-proc "$out/backend-$1" bash "$W/h2h-paired/pf/pf-run" \
    "$W/pytorch-fugaku-b5879f2.sif" python -u /cross/h2h-rsalgo/test_backend_mpi.py
  echo "backend ag=$1 rs=$2 exit=$?" >> "$out/summary.txt"
done
cd "$W/h2h-scale"
H2H_CASES='container%auto' H2H_TRAINER=tp_train.py H2H_TP=4 H2H_TP_SP=1 H2H_TP_CHECK=1 H2H_MODEL=tiny \
  H2H_SEQ=256 H2H_RPN=4 H2H_STEPS=4 H2H_WARMUP=1 H2H_CASE_TIMEOUT=900 bash ./run.sh
echo "sp check exit=$? (results.$PJM_JOBID)" >> "$out/summary.txt"
