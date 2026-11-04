"""Tensor parallel x FSDP2 for transformers' Llama on Fugaku CPUs.

With 4 ranks x 12 threads per node and ``tp=4``, a node is one tensor-parallel
group, so tensor-parallel traffic stays in the node; FSDP2 shards across
nodes. This is the route that fits 30B-class models at 4 ranks per node
(DeepSpeed ZeRO-3 alone cannot: the gathered vocabulary embedding, its
gradient and copies exceed a rank's share of the node).

- q/k/v, gate/up column-parallel; o, down row-parallel.
- Embedding, lm_head and the loss vocab-parallel (``loss_parallel``).
- ``sp=True``: the residual stream, norms and checkpointed layer inputs are
  sharded along the sequence inside the tensor-parallel group.
- ``fully_shard`` per decoder layer, embedding and lm_head, then the root.

    mesh = init_device_mesh("cpu", (world // 4, 4), mesh_dim_names=("dp", "tp"))
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    parallelize_llama(model, mesh, sp=True)
    materialize_llama(model, config)
    loss = model(ids)                         # forward is the vocab-parallel loss
    with loss_parallel():
        loss.backward()

Every rank of a tensor-parallel group must get the same batch.
"""
import types

import torch
import torch.nn.functional as F

__all__ = ["parallelize_llama", "materialize_llama", "lm_loss"]


def lm_loss(model, ids):
    """Mean next-token cross-entropy with vocab-sharded logits (the parallel model's forward)."""
    from torch.distributed.tensor.parallel import loss_parallel

    h = model.model(input_ids=ids).last_hidden_state
    logits = model.lm_head(h)  # DTensor, Shard(-1): [b, s, vocab / tp] per rank
    labels = torch.cat([ids[:, 1:], ids.new_full((ids.shape[0], 1), -100)], 1)
    with loss_parallel():
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)


def parallelize_llama(model, mesh, sp=True, checkpointing=True):
    """Apply the plan in place on a ``LlamaForCausalLM`` (meta or materialized) over a 2-D mesh
    with dims ("dp", "tp"). Untied embeddings only. Returns the model; ``model(ids)`` is the loss."""
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                                   SequenceParallel, parallelize_module)

    tp, dp = mesh["tp"], mesh["dp"]
    # src_data_rank=None: each rank keeps its own chunk of the (identical, or meta) full tensor
    # instead of a scatter from rank 0 (torch_fugaku's fjmpi backend has no scatter).
    # sp: the embedding output stays whole, so transformers' mask and positions see the full
    # sequence; layer 0 shards it on entry.
    seq = Shard(1) if sp else Replicate()
    parallelize_module(model, tp, {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),  # vocab-sharded rows
        "lm_head": ColwiseParallel(input_layouts=seq, output_layouts=Shard(-1), use_local_output=False),
        **({"model.norm": SequenceParallel(use_local_output=True)} if sp else {}),
    }, src_data_rank=None)
    layer_plan = {
        "self_attn.q_proj": ColwiseParallel(), "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(), "self_attn.o_proj": RowwiseParallel(output_layouts=seq),
        "mlp.gate_proj": ColwiseParallel(), "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(output_layouts=seq),
    }
    if sp:
        layer_plan.update({
            "input_layernorm": SequenceParallel(use_local_output=True),
            "post_attention_layernorm": SequenceParallel(use_local_output=True),
            "self_attn": PrepareModuleInput(input_kwarg_layouts={"hidden_states": Shard(1)},
                                            desired_input_kwarg_layouts={"hidden_states": Replicate()},
                                            use_local_output=True),
            "mlp": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),),
                                      use_local_output=True),
        })
        parallelize_module(model.model.layers[0], tp, PrepareModuleInput(
            input_layouts=(Replicate(),), desired_input_layouts=(Shard(1),), use_local_output=True))
    for layer in model.model.layers:
        parallelize_module(layer, tp, layer_plan, src_data_rank=None)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Own FSDP units for the embedding and lm_head, so each is gathered only while it is used.
    for m in (*model.model.layers, model.model.embed_tokens, model.lm_head):
        fully_shard(m, mesh=dp)
    model.forward = types.MethodType(lm_loss, model)  # FSDP's root hooks run on model(...)
    fully_shard(model, mesh=dp)
    return model


def materialize_llama(model, config):
    """Allocate a meta-built, parallelized model's shards and initialize them as transformers'
    Llama does (N(0, initializer_range), norms 1); rebuild the rotary buffers."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    model.to_empty(device="cpu")
    with torch.no_grad():
        for name, prm in model.named_parameters():
            (prm.fill_(1.0) if "norm" in name else prm.normal_(0.0, config.initializer_range))
    model.model.rotary_emb = LlamaRotaryEmbedding(config=config)
    return model
