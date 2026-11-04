#!/bin/bash
# Legacy DeepSpeedFugaku.sif (Fugaku-LLM Megatron-DeepSpeed, Fujitsu PyTorch 1.13): can it train
# Megatron GPT-125M (12 layers, 768 hidden, 12 heads; DP only) on 1 node? Settings follow
# DeepSpeedFugaku/0.125b_pp1_tp1_dp1_outer.sh + inner.sh, without --save, seq 1024 as the sweep notes.
W=/vol0402/data/ra000009/u10035/agent/work/legacy-sif
SIF=$W/DeepSpeedFugaku.sif
out=$W/smoke.$PJM_JOBID; mkdir -p $out
D=/work/DeepSpeedFugaku
args="--num-layers 12 --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 \
 --seq-length 1024 --max-position-embeddings 1024 --train-iters 6 --lr-decay-iters 320000 \
 --data-path $D/dataset/codeparrot_content_document --tokenizer-type JapaneseSentencePiece \
 --vocab-file $D/tokenizer/models/ver2/code20k_en40k_ja60k.ver2.1.model --data-impl mmap --split 949,50,1 \
 --distributed-backend mpi --init-method-std 0.013 --lr 6e-4 --min-lr 1.0e-6 --lr-decay-style cosine \
 --weight-decay 1e-2 --cpu-optimizer --cpu-torch-adam --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 \
 --lr-warmup-fraction .01 --checkpoint-activations --log-interval 1 --save-interval 10000 --eval-interval 1000 \
 --eval-iters 0 --no-cuda --use-cpu-initialization --num-workers 0 --no-load-rng \
 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --use-flush-denormal"
export SINGULARITYENV_MASTER_ADDR=localhost SINGULARITYENV_MASTER_PORT=$((10000 + PJM_JOBID % 50000))
cd $W
# The image expects the host's Fujitsu runtime (tcsds-1.2.36) and, for MPI across ranks, the
# host's Tofu/PMIx libraries: the same binds as pytorch-fugaku's pf-run, hostlib from pf-setup.
export PF_HOSTLIB=$W/.pf-hostlib.$PJM_JOBID
bash /vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-paired/pf/pf-setup 4 > $out/pf-setup.txt 2>&1
B=/opt/FJSVxtclanga,/var/opt/FJSVtcs,/run,/usr/lib/FJSVtcs,/etc/opt/FJSVtcs,/sys/devices/system/cpu,/sys/fs/cgroup
B=$B,$PF_HOSTLIB:/hostlib,$W:/work
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/FJSVxtclanga/tcsds-1.2.36/lib64:/hostlib
singularity run -B $B $SIF python -c "import torch, deepspeed, sys; print(sys.version); print(torch.__version__, deepspeed.__version__, torch.distributed.is_mpi_available(), torch.get_num_threads())" > $out/a-import.txt 2>&1
echo "a exit=$?" >> $out/progress.txt
# The image ships upstream Megatron-DeepSpeed only; run Fugaku-LLM's bundle (extracted to $W) from its own directory.
# tcmalloc (as Fugaku-LLM's inner.sh) needs libunwind.so.8: find it inside the image.
unw=$(singularity run -B $B $SIF sh -c "find / -xdev -name libunwind.so.8 2>/dev/null | head -1")
[[ -z "$unw" && -e /usr/lib64/libunwind.so.8 ]] && { cp -L /usr/lib64/libunwind.so.8 $PF_HOSTLIB/; unw=/hostlib/libunwind.so.8; }
echo "libunwind: ${unw:-none}" >> $out/progress.txt
if [[ -n "$unw" ]]; then
  export SINGULARITYENV_LD_LIBRARY_PATH="$(dirname "$unw"):$SINGULARITYENV_LD_LIBRARY_PATH"
  export SINGULARITYENV_LD_PRELOAD=/local/fcc/inst/other/lib/libtcmalloc.so
  if singularity run -B $B $SIF python -c "import torch" >> $out/progress.txt 2>&1; then
    echo "tcmalloc: on" >> $out/progress.txt
  else
    unset SINGULARITYENV_LD_PRELOAD
    echo "tcmalloc: off (preload failed)" >> $out/progress.txt
  fi
else
  echo "tcmalloc: off (no libunwind)" >> $out/progress.txt
fi
# b: 1 rank x 48 threads, as the 0.125b script (global batch 1)
SINGULARITYENV_OMP_NUM_THREADS=48 SINGULARITYENV_WORLD_SIZE=1 timeout 600 mpiexec -n 1 -std-proc $out/b \
  singularity run -B $B $SIF bash -c "cd $D && exec numactl -m 4-7 -N 4-7 python $D/pretrain_gpt.py $args --global-batch-size 1"
echo "b exit=$?" >> $out/progress.txt
# c: 4 ranks x 12 threads (the sweep notes best DP layout), global batch 4
SINGULARITYENV_OMP_NUM_THREADS=12 SINGULARITYENV_WORLD_SIZE=4 timeout 600 mpiexec -n 4 -std-proc $out/c \
  singularity run -B $B $SIF bash -c "cd $D && exec python $D/pretrain_gpt.py $args --global-batch-size 4"
echo "c exit=$?" >> $out/progress.txt
