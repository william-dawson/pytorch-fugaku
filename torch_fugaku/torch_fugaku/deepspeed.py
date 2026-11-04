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
"""
import importlib.abc
import importlib.machinery
import math
import os
import sys

__all__ = ["fast_zero3", "install"]

_state = {"patched": False}


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


class _AfterStage3Import(importlib.abc.MetaPathFinder):
    _name = "deepspeed.runtime.zero.stage3"

    def find_spec(self, name, path, target=None):
        if name != self._name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            sys.modules[self._name] = module  # fast_zero3 imports it
            fast_zero3()

        spec.loader.exec_module = exec_module
        return spec


def install():
    """Apply fast_zero3 now if ZeRO-3 is loaded, else when it is first imported."""
    if os.environ.get("TORCH_FUGAKU_FAST_ZERO", "1") == "0" or _state["patched"]:
        return
    if _AfterStage3Import._name in sys.modules:
        fast_zero3()
    elif not any(isinstance(f, _AfterStage3Import) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterStage3Import())
