"""DeepSpeed ZeRO-3 without its per-rank Python loops.

DeepSpeed's ``reduce_scatter_coalesced`` slices every gradient into world-size
chunks and concatenates them; ``AllGatherCoalescedHandle.wait`` narrows every
gathered parameter world-size times and concatenates again. Both cost
O(ranks x tensors) small ops per step: on Fugaku 0.8 s per step at 64 ranks and
4.3 s at 512 for a 1B model, growing linearly. ``fast_zero3()`` replaces both
with one strided copy per tensor into a ``[world, chunk]`` buffer of DeepSpeed's
own layout, so results are bit-identical.

In the pytorch-fugaku image (``TORCH_FUGAKU=1``) this is applied when DeepSpeed's
ZeRO-3 module is first imported; ``TORCH_FUGAKU_FAST_ZERO=0`` turns it off.

``no_jit_ops()`` (also on import of ``deepspeed``, unless ``TORCH_FUGAKU_DS_OPS=1``) marks
DeepSpeed's C++ ops unavailable, so it never tries to JIT-build them (its CPU accelerator wants
ninja, which the image does not ship) and its collectives go through torch.distributed, i.e.
the fjmpi backend. Every recorded measurement ran this way.
"""
import importlib.abc
import importlib.machinery
import math
import os
import sys

__all__ = ["fast_zero3", "no_jit_ops", "install"]

_state = {"patched": False, "no_jit": False}


def no_jit_ops():
    """Mark every DeepSpeed op incompatible, and make an explicit load an error. Idempotent."""
    if _state["no_jit"]:
        return
    import deepspeed.ops
    from deepspeed.ops.op_builder.all_ops import ALL_OPS
    from deepspeed.ops.op_builder.builder import OpBuilder

    deepspeed.ops.__compatible_ops__ = dict.fromkeys(ALL_OPS, False)

    def no_jit(self, *args, **kwargs):
        raise RuntimeError(f"DeepSpeed op {self.name}: torch_fugaku disabled JIT ops (TORCH_FUGAKU_DS_OPS=1 enables)")

    OpBuilder.load = OpBuilder.jit_load = no_jit
    _state["no_jit"] = True


def fast_zero3():
    """Patch DeepSpeed (0.16+) in place. Idempotent; needs deepspeed importable."""
    if _state["patched"]:
        return
    import torch
    import torch.distributed as dist

    import deepspeed.runtime.comm.coalesced_collectives as cc
    import deepspeed.runtime.zero.stage3 as stage3
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.zero import partition_parameters as pp

    @torch.no_grad()
    def reduce_scatter_coalesced(tensors, group=None):
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        sizes = [math.ceil(t.numel() / world) for t in tensors]
        # [world, sum(sizes)]: row r holds every tensor's r-th chunk (padded), as DeepSpeed's cat.
        grid = torch.empty(world, sum(sizes), dtype=tensors[0].dtype, device=tensors[0].device)
        off = 0
        for t, c in zip(tensors, sizes):
            v, n = t.reshape(-1), t.numel()
            full = n // c if c else 0
            grid[:full, off:off + c].copy_(v[:full * c].view(full, c))
            if full < world and n > full * c:
                grid[full, off:off + n - full * c].copy_(v[full * c:])
            off += c
        grid.div_(world)  # pre-divide, as DeepSpeed
        out = torch.empty(off, dtype=grid.dtype, device=grid.device)
        cc._torch_reduce_scatter_fn(grid.view(-1), out, group=group)
        result, off = [], 0
        for t, c in zip(tensors, sizes):
            result.append(out.narrow(0, off, max(0, min(c, t.numel() - rank * c))))
            off += c
        return result

    plain_wait = pp.AllGatherCoalescedHandle.wait

    def wait(self, handle_dependency=True):
        if self.complete:
            return
        base = self.partitions[0]._base if self.partitions else None
        if (self.quantization or self.use_secondary_tensor or base is None
                or not get_accelerator().is_synchronized_device()):
            return plain_wait(self, handle_dependency)
        self.allgather_handle.wait()
        size = self.partitions[0].numel()
        grid = base.view(-1)[:size * self.world_size].view(self.world_size, size)
        off = 0
        for param in self.params:
            s = param.ds_tensor.ds_numel
            # rank-major shards of this parameter, then drop the padding
            param.data = grid[:, off:off + s].reshape(-1)[:param.ds_numel].view(param.ds_shape).to(param.ds_tensor.dtype)
            param.ds_status = pp.ZeroParamStatus.AVAILABLE
            off += s
        self.complete = True

    cc.reduce_scatter_coalesced = stage3.reduce_scatter_coalesced = reduce_scatter_coalesced
    pp.AllGatherCoalescedHandle.wait = wait
    _state["patched"] = True


class _AfterImport(importlib.abc.MetaPathFinder):
    """Run ``then`` once, right after module ``name`` has executed."""

    def __init__(self, name, then):
        self.name, self.then = name, then

    def find_spec(self, name, path, target=None):
        if name != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            sys.modules[self.name] = module
            self.then()

        spec.loader.exec_module = exec_module
        return spec


def _when_imported(name, then):
    if name in sys.modules:
        then()
    elif not any(isinstance(f, _AfterImport) and f.name == name for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterImport(name, then))


def install():
    """no_jit_ops when deepspeed is imported, fast_zero3 when ZeRO-3 is (or now, if already)."""
    if os.environ.get("TORCH_FUGAKU_DS_OPS", "0") != "1":
        _when_imported("deepspeed", no_jit_ops)
    if os.environ.get("TORCH_FUGAKU_FAST_ZERO", "1") != "0":
        _when_imported("deepspeed.runtime.zero.stage3", fast_zero3)
