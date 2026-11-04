#!/bin/bash
# 2 nodes: validate the repository's torch_fugaku (Python modules + c10d_mpi.cpp) in the image:
# test_nn.py, test_backend_mpi.py with rings forced, examples/tp_train.py --check --sp, and
# examples/llama_train.py (ZeRO-3) checking that torch_fugaku.deepspeed patched itself.
set -uo pipefail
here=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-rsalgo
W=$(dirname "$here")
cd "$here"
out=$here/prep3.$PJM_JOBID; mkdir -p "$out"
OVERLAY=container-next bash build_container.sh > "$out/build.txt" 2>&1; echo "build exit=$?" >> "$out/summary.txt"
SIF=$W/pytorch-fugaku-b5879f2.sif
export PF_HOSTLIB="$here/.pf-hostlib.${PJM_JOBID}"
bash "$W/h2h-paired/pf/pf-setup" 4
set -a; source "$PF_HOSTLIB/pf.env"; set +a
export PATH=/cross/h2h-paired/site310/bin:$PATH PF_BIND="$W:/cross:rw" PYTHONPATH=/cross/h2h-rsalgo/container-next:/cross/h2h-paired/site310 \
       OMP_NUM_THREADS=12 DS_ACCELERATOR=cpu TORCH_EXTENSIONS_DIR=/cross/h2h-rsalgo/.torch_ext \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MIMALLOC_PURGE_DELAY=0
singularity exec -B "$PF_BIND" "$SIF" env PYTHONPATH=$PYTHONPATH python /cross/h2h-rsalgo/tests/test_nn.py > "$out/test_nn.txt" 2>&1
echo "test_nn exit=$?" >> "$out/summary.txt"
run() {  # name, command...
  local name=$1; shift
  timeout 900 mpiexec -n "$WORLD_SIZE" -std-proc "$out/$name" bash "$W/h2h-paired/pf/pf-run" "$SIF" "$@"
  echo "$name exit=$?" >> "$out/summary.txt"
}
TORCH_FUGAKU_RS_ALGO=ring TORCH_FUGAKU_AG_ALGO=ring run backend python -u /cross/h2h-rsalgo/tests/test_backend_mpi.py
run tpcheck python -u /cross/h2h-rsalgo/examples/tp_train.py --check --model tiny --tp 4 --sp --seq 256 \
  --steps 4 --warmup 1 --checkpointing
run zero3 python -u /cross/h2h-rsalgo/examples/llama_train.py --backend fjmpi --model gpt-125m --seq 1024 \
  --zero 3 --low-mem --checkpointing --attn sdpa_causal --loss-chunk 256 --zero-init --steps 3 --warmup 1
