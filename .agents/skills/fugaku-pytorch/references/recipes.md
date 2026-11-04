# Recipes

All of these run the image through `pf-setup` (once per job) and `pf-run` (once per rank under
Fujitsu `mpiexec`); see the repository README for pulling the image.

DeepSpeed 0.19.8, transformers 4.57.6 and accelerate 1.15.0 are in the image, with
`DS_ACCELERATOR=cpu`. torch_fugaku turns DeepSpeed's C++ ops off on import, so nothing is
JIT-built; `TORCH_FUGAKU_DS_OPS=1` restores DeepSpeed's default. Bind your code and data with
`PF_BIND`, for example `export PF_BIND=/vol0X0X/data/<group>:/work`.

## DeepSpeed ZeRO-3 (1-3B models)

```sh
#PJM -L "node=16" -L "freq=2000,eco_state=0" -L "elapse=01:00:00"
#PJM --mpi "max-proc-per-node=4"
#PJM --llio cn-cache-size=384Mi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
export OMP_NUM_THREADS=12 PF_FJBLAS=1
pytorch-fugaku/fugaku/pf-setup 4
mpiexec -n 64 pytorch-fugaku/fugaku/pf-run pytorch-fugaku.sif python \
  /opt/pytorch-fugaku/examples/llama_train.py --backend fjmpi --model 1b --seq 2048 --zero 3 \
  --low-mem --checkpointing --attn sdpa_causal --loss-chunk 256 --zero-init --steps 20
```

In your own script:

```python
import torch_fugaku, deepspeed
from torch_fugaku.nn import chunked_lm_loss
deepspeed.init_distributed(dist_backend="fjmpi", init_method="fjmpi://")  # no TCP rendezvous
# ZeRO-3 config: keep DeepSpeed's live parameters small at 4 ranks per node, e.g.
# stage3_max_live_parameters=1e8, stage3_prefetch_bucket_size=2e7, reduce_bucket_size=5e7
model = AutoModelForCausalLM.from_config(config)      # partitioned under zero.Init
chunked_lm_loss(model, 256)                            # no [tokens, vocab] logits
```

The ZeRO-3 per-rank-loop patch applies itself; nothing to call.

## Tensor parallel × FSDP2 (10-30B+)

Tensor parallel 4 maps onto the 4 ranks of a node; FSDP2 shards across nodes. Every rank of a
node (a tensor-parallel group) must get the same batch; nodes get different ones.

```sh
#PJM -L "node=64" -L "freq=2000,eco_state=0" -L "elapse=02:00:00"
#PJM --mpi "max-proc-per-node=4"
#PJM --llio cn-cache-size=384Mi
export OMP_NUM_THREADS=12 PF_FJBLAS=1 MIMALLOC_PURGE_DELAY=0   # memory is tight at 30B
pytorch-fugaku/fugaku/pf-setup 4
mpiexec -n 256 pytorch-fugaku/fugaku/pf-run pytorch-fugaku.sif python \
  /opt/pytorch-fugaku/examples/tp_train.py --model 30b --tp 4 --sp --checkpointing --seq 2048 --steps 10
```

In your own script:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import loss_parallel
from torch_fugaku.parallel import parallelize_llama, materialize_llama
dist.init_process_group("fjmpi", init_method="fjmpi://")
mesh = init_device_mesh("cpu", (world // 4, 4), mesh_dim_names=("dp", "tp"))
with torch.device("meta"):
    model = LlamaForCausalLM(config)        # tie_word_embeddings=False
parallelize_llama(model, mesh, sp=True)
materialize_llama(model, config)
opt = torch.optim.AdamW(model.parameters(), lr=..., foreach=False)  # params live on two meshes
loss = model(ids)                           # forward is the vocab-parallel loss
with loss_parallel():
    loss.backward()
```

`examples/tp_train.py --check --model tiny` verifies the plan against an unsharded copy (2+
nodes).

## Node shapes

`node=N` in resource groups of up to 384 nodes gives a **non-contiguous** allocation. Ring
collectives, which FSDP's one-rank-per-node groups use, then vary by up to 2× between
allocations. A torus shape keeps ring neighbours adjacent:

```sh
#PJM -L "node=4x6x4:torus"     # 96 nodes; Tofu units are 2x3x2 = 12 nodes
```

Groups above 384 nodes allow only `:torus`.
