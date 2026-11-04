"""Llama training with PyTorch-native 2-D parallelism on Fugaku CPUs: tensor parallel (DTensor)
inside each node, FSDP2 (fully_shard) across nodes. With 4 ranks x 12 threads per node and
--tp 4, a node holds one tensor-parallel group, so tensor-parallel traffic never leaves the node.

- Each decoder layer: q/k/v, gate/up column-parallel; o, down row-parallel.
- Embedding, lm_head and the loss are vocab-parallel (loss_parallel), so no rank ever holds the
  [vocab, hidden] weight or the [tokens, vocab] logits.
- fully_shard over the data-parallel mesh dimension, per layer and for the rest; activation
  checkpointing per layer.
- The model is built on the meta device and materialized as shards (--check builds it whole).
The plan is torch_fugaku.parallel (parallelize_llama, materialize_llama).

Every rank of a tensor-parallel group gets the same batch; data-parallel ranks get different ones.
--check: tiny model, compared step by step against an unsharded copy trained on the same global
batch on every rank.

Prints the same JSON events as llama_train.py (setup, mem, phase, step, fom).
"""
import argparse
import copy
import json
import os
import resource
import time

T_START = time.perf_counter()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODELS = {
    "tiny": dict(hidden_size=256, intermediate_size=704, num_hidden_layers=2, num_attention_heads=8,
                 num_key_value_heads=4, vocab_size=8192),
    "1b": dict(hidden_size=2048, intermediate_size=8192, num_hidden_layers=16, num_attention_heads=32,
               num_key_value_heads=8, vocab_size=128256),
    # The legacy SIF's 30B shape (48 layers, hidden 6912, 72 heads, GPT-2 vocabulary) as a Llama:
    # full multi-head attention and a SwiGLU MLP of 2/3 x 4h, so each layer has GPT's 12h^2 weights.
    "30b": dict(hidden_size=6912, intermediate_size=18432, num_hidden_layers=48, num_attention_heads=72,
                num_key_value_heads=72, vocab_size=50304),
}


def log(rank, **kw):
    if rank == 0:
        print(json.dumps(kw), flush=True)


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    return 0.0


def hwm_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def node_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return 0.0


def phase(step, name, start, rss0):
    # every rank logs its own phases (mem_probe.py takes the max over ranks)
    print(json.dumps(dict(event="phase", step=step, phase=name, wall_s=time.perf_counter() - start,
                          rss_start_gb=rss0, rss_end_gb=rss_gb(), rss_peak_gb=hwm_gb())), flush=True)


def census(model, opt, step):
    """Bytes of live tensor storage on this rank by owner: parameters, gradients, optimizer
    state, everything else the garbage collector can see (unique storages), and VmRSS."""
    import gc

    def local(t):
        return t.to_local() if hasattr(t, "to_local") else t

    seen, by = set(), {"param": 0, "grad": 0, "optim": 0, "other": 0}

    def add(t, kind):
        t = local(t)
        if not isinstance(t, torch.Tensor) or t.device.type == "meta":
            return
        st = t.untyped_storage()
        if st.data_ptr() in seen or st.nbytes() == 0:
            return
        seen.add(st.data_ptr())
        by[kind] += st.nbytes()

    for prm in model.parameters():
        add(prm, "param")
        if prm.grad is not None:
            add(prm.grad, "grad")
    for state in opt.state.values():
        for v in state.values():
            add(v, "optim")
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                add(obj, "other")
        except Exception:
            pass
    print(json.dumps(dict(event="census", step=step, rss_gb=rss_gb(), **{k: v / 2**30 for k, v in by.items()})),
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny", choices=sorted(MODELS))
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--backend", default="fjmpi")
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--sp", action="store_true", help="sequence parallel within the tensor-parallel group")
    p.add_argument("--census", action="store_true", help="live-tensor census after each of the first steps")
    p.add_argument("--check", action="store_true", help="compare against an unsharded copy (small models)")
    p.add_argument("--dtype", default="fp32")  # accepted for rank.sh compatibility; fp32 only
    p.add_argument("--zero", type=int, default=3)  # accepted for rank.sh compatibility
    p.add_argument("--low-mem", action="store_true")
    p.add_argument("--loss-chunk", type=int, default=0)
    p.add_argument("--dropout", type=float, default=None)
    a, _ = p.parse_known_args()

    if a.backend == "fjmpi":
        import torch_fugaku

        torch_fugaku.register_backend()
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import loss_parallel
    from transformers import LlamaConfig, LlamaForCausalLM
    from torch_fugaku.parallel import materialize_llama, parallelize_llama

    t0 = time.perf_counter()
    if not dist.is_initialized():
        # pf-run sets RANK/WORLD_SIZE/MASTER_ADDR; a plain mpirun sets only its own variables.
        env = os.environ
        env.setdefault("RANK", env.get("PMIX_RANK", env.get("OMPI_COMM_WORLD_RANK", "0")))
        env.setdefault("WORLD_SIZE", env.get("OMPI_COMM_WORLD_SIZE", "1"))
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29513")
        # fjmpi:// (torch_fugaku): no TCP rendezvous, which failed on Fugaku from ~1000 ranks.
        dist.init_process_group(a.backend, init_method="fjmpi://" if a.backend == "fjmpi" else None)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world % a.tp == 0, (world, a.tp)
    dp_size = world // a.tp
    mesh = init_device_mesh("cpu", (dp_size, a.tp), mesh_dim_names=("dp", "tp"))
    dp_rank = mesh["dp"].get_local_rank()
    t_pg = time.perf_counter() - t0

    spec = dict(MODELS[a.model])
    if a.layers:
        spec["num_hidden_layers"] = a.layers
    cfg = LlamaConfig(max_position_embeddings=a.seq, attn_implementation="sdpa" if a.attn == "sdpa_causal" else a.attn,
                      tie_word_embeddings=False, **spec)
    t0 = time.perf_counter()
    torch.manual_seed(0)
    ref = None
    if a.check:
        model = LlamaForCausalLM(cfg)  # whole, same on every rank (same seed)
        ref = copy.deepcopy(model)
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
        parallelize_llama(model, mesh, sp=a.sp, checkpointing=a.checkpointing)
    else:
        with torch.device("meta"):
            model = LlamaForCausalLM(cfg)
        parallelize_llama(model, mesh, sp=a.sp, checkpointing=a.checkpointing)
        materialize_llama(model, cfg)
    model.train()
    nparams = sum(prm.numel() for prm in model.parameters())  # DTensor numel is the global size
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if a.check else 1e-4, weight_decay=0.1,
                            betas=(0.9, 0.95), foreach=False)  # parameters live on two meshes
    t_model = time.perf_counter() - t0
    print(json.dumps(dict(event="mem", rank=rank, phase="model built", step=None, rss_gb=rss_gb(),
                          node_avail_gb=node_avail_gb())), flush=True)
    log(rank, event="setup", trainer="tp_train", model=a.model, params=nparams, world=world, tp=a.tp, dp=dp_size, sp=a.sp,
        layers=cfg.num_hidden_layers, seq=a.seq, micro_batch=a.micro_batch, backend=a.backend,
        threads=torch.get_num_threads(), t_process_group_s=round(t_pg, 2), t_model_s=round(t_model, 2))

    hook = globals().get
    times, worst = [], 0.0
    for step in range(a.steps):
        hook("h2h_step_start", lambda s: None)(step)
        # one batch per data-parallel rank; every rank can rebuild every batch (for --check)
        batches = [torch.randint(0, cfg.vocab_size, (a.micro_batch, a.seq),
                                 generator=torch.Generator().manual_seed(1000 * step + d)) for d in range(dp_size)]
        ids = batches[dp_rank]
        t0 = time.perf_counter()
        r0 = rss_gb()
        loss = model(ids)
        phase(step, "forward", t0, r0) if step < 2 else None
        t1, r1 = time.perf_counter(), rss_gb()
        with loss_parallel():
            loss.backward()
        phase(step, "backward", t1, r1) if step < 2 else None
        t2, r2 = time.perf_counter(), rss_gb()
        opt.step()
        opt.zero_grad(set_to_none=True)
        phase(step, "step", t2, r2) if step < 2 else None
        if a.census and step < 3:
            census(model, opt, step)
        lt = loss.detach().full_tensor() if hasattr(loss, "full_tensor") else loss.detach()
        lt = lt.float().clone()
        dist.all_reduce(lt)
        loss_val = lt.item() / world  # every TP rank holds its DP rank's loss
        dt = time.perf_counter() - t0
        times.append(dt)
        hook("h2h_record_step", lambda *x: None)(step, dt, loss_val)
        if ref is not None:
            ref_losses = []
            for b in batches:  # the global batch, averaged as FSDP's data-parallel mean
                lr_ = lm_loss_plain(ref, b) / dp_size
                lr_.backward()
                ref_losses.append(lr_.item() * dp_size)
            ref_opt.step()
            ref_opt.zero_grad(set_to_none=True)
            ref_loss = sum(ref_losses) / dp_size
            worst = max(worst, abs(ref_loss - loss_val) / abs(ref_loss))
            log(rank, event="step", step=step, loss=round(loss_val, 6), ref_loss=round(ref_loss, 6),
                rel_err=abs(ref_loss - loss_val) / abs(ref_loss), time_s=round(dt, 3))
        else:
            log(rank, event="step", step=step, loss=round(loss_val, 6), time_s=round(dt, 3))

    if ref is not None:
        log(rank, event="check", worst_rel_err=worst, ok=worst < 1e-4)
    rss = torch.tensor([hwm_gb()])
    dist.all_reduce(rss, op=dist.ReduceOp.MAX)
    steady = times[a.warmup:] or times
    tokens = a.micro_batch * a.seq * dp_size  # one batch per tensor-parallel group
    tps = tokens * len(steady) / sum(steady)
    flop_per_token = 6 * nparams + 12 * cfg.num_hidden_layers * cfg.hidden_size * a.seq
    nodes = int(os.environ.get("NNODES", os.environ.get("PJM_NODE", "1")))
    log(rank, event="fom", tokens_per_s=round(tps, 1), tokens_per_s_per_node=round(tps / nodes, 2),
        tflops_per_node=round(tps * flop_per_token / nodes / 1e12, 3), nodes=nodes,
        first_step_s=round(times[0], 2), steady_step_s=round(sum(steady) / len(steady), 3),
        peak_rss_gb_max_rank=round(rss.item(), 2))


def lm_loss_plain(model, ids):
    logits = model(input_ids=ids).logits
    labels = torch.cat([ids[:, 1:], ids.new_full((ids.shape[0], 1), -100)], 1)
    return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=-100)


if __name__ == "__main__":
    main()
