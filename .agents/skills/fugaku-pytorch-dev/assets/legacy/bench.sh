#!/bin/bash
# Legacy stack benchmark: Fugaku-LLM's Megatron-DeepSpeed (DeepSpeedFugaku bundle, extracted here,
# with its per-step profiler made opt-in) in DeepSpeedFugaku.sif (Fujitsu PyTorch 1.13, DeepSpeed 0.8).
# Data-parallel only (TP = PP = 1), no gradient accumulation: global batch = ranks x micro-batch.
#   L_MODEL   gpt-125m (12 x 768, 12 heads) | gpt-1.3b (24 x 2048, 32 heads)   -- the sweep notes' shapes
#   L_RPN     ranks per node (4: 12 threads each; 1: 48 threads, numactl over CMGs 4-7 as their scripts)
#   L_MBS     micro-batch per rank;  L_SEQ (default 1024);  L_ITERS (default 6)
#   L_TP      tensor-parallel size (default 1; Fugaku-LLM ran 1.3B as TP 4 over 4 nodes x 1 rank)
#   L_DROPOUT hidden and attention dropout (default: Megatron's 0.1)
#   L_MALLOC  mimalloc: preload the LLVM work's mimalloc (h2h-alloc) as the stand-in for their
#             tcmalloc; without a caching allocator GPT-1.3B runs out of memory (job 53487634).
# Not reproducible here: their LD_PRELOAD=libtcmalloc.so (needs libunwind.so.8, absent in the image
# and on compute nodes) and freq=2200. Per-iteration times come from Megatron's own log line.
set -uo pipefail
W=/vol0402/data/ra000009/u10035/agent/work/legacy-sif
SIF=$W/DeepSpeedFugaku.sif
D=/work/DeepSpeedFugaku
model=${L_MODEL:?} rpn=${L_RPN:?} mbs=${L_MBS:?} seq=${L_SEQ:-1024} iters=${L_ITERS:-6} tp=${L_TP:-1}
nodes=${PJM_NODE:?}
ranks=$((nodes * rpn))
out=$W/bench.$PJM_JOBID; mkdir -p $out
case "$model" in
  gpt-125m) shape="--num-layers 12 --hidden-size 768 --num-attention-heads 12" ;;
  gpt-1.3b) shape="--num-layers 24 --hidden-size 2048 --num-attention-heads 32" ;;
  *) echo "unknown model $model" >&2; exit 2 ;;
esac
args="$shape --micro-batch-size $mbs --global-batch-size $((ranks / tp * mbs)) \
 --seq-length $seq --max-position-embeddings $seq --train-iters $iters --lr-decay-iters 320000 \
 --data-path $D/dataset/codeparrot_content_document --tokenizer-type JapaneseSentencePiece \
 --vocab-file $D/tokenizer/models/ver2/code20k_en40k_ja60k.ver2.1.model --data-impl mmap --split 949,50,1 \
 --distributed-backend mpi --init-method-std 0.013 --lr 6e-4 --min-lr 1.0e-6 --lr-decay-style cosine \
 --weight-decay 1e-2 --cpu-optimizer --cpu-torch-adam --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 \
 --lr-warmup-fraction .01 --checkpoint-activations --log-interval 1 --save-interval 10000 --eval-interval 1000 \
 --eval-iters 0 --no-cuda --use-cpu-initialization --num-workers 0 --no-load-rng \
 --tensor-model-parallel-size $tp --pipeline-model-parallel-size 1 --use-flush-denormal"
[[ -n "${L_DROPOUT:-}" ]] && args="$args --hidden-dropout $L_DROPOUT --attention-dropout $L_DROPOUT"
printf 'model=%s nodes=%s rpn=%s ranks=%s tp=%s mbs=%s seq=%s iters=%s malloc=%s dropout=%s\n' $model $nodes $rpn $ranks $tp $mbs $seq $iters "${L_MALLOC:-glibc}" "${L_DROPOUT:-0.1}" > $out/config.txt
# Host Fujitsu runtime (the image expects tcsds-1.2.36) and Tofu/PMIx libraries: pf-run's binds.
export PF_HOSTLIB=$W/.pf-hostlib.$PJM_JOBID
bash /vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-paired/pf/pf-setup $rpn > $out/pf-setup.txt 2>&1
B=/opt/FJSVxtclanga,/var/opt/FJSVtcs,/run,/usr/lib/FJSVtcs,/etc/opt/FJSVtcs,/sys/devices/system/cpu,/sys/fs/cgroup
B=$B,$PF_HOSTLIB:/hostlib,$W:/work
if [[ "${L_MALLOC:-}" == mimalloc ]]; then
  B=$B,/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-alloc/build:/alloc
  export SINGULARITYENV_LD_PRELOAD=/alloc/libmimalloc.so.2.2
fi
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/FJSVxtclanga/tcsds-1.2.36/lib64:/hostlib
export SINGULARITYENV_OMP_NUM_THREADS=$((48 / rpn)) SINGULARITYENV_WORLD_SIZE=$ranks
export SINGULARITYENV_MASTER_ADDR=localhost SINGULARITYENV_MASTER_PORT=$((10000 + PJM_JOBID % 50000))
pin=; (( rpn == 1 )) && pin="numactl -m 4-7 -N 4-7"
cd $W
timeout ${L_TIMEOUT:-1500} mpiexec -n $ranks -std-proc $out/rank \
  singularity run -B $B $SIF bash -c "cd $D && exec $pin python $D/pretrain_gpt.py $args"
echo "exit=$?" > $out/exit.txt
# Megatron logs iterations on the last rank.
grep -hE "iteration +[0-9]+/" $out/rank.*.$((ranks - 1)) | \
  sed -E 's/.*iteration +([0-9]+)\/.*per iteration \(ms\): ([0-9.]+).*lm loss: ([0-9.E+-]+).*/\1 \2 \3/' > $out/iterations.txt
