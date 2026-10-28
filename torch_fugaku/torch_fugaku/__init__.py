"""PyTorch on Fugaku (A64FX) with the stock aarch64 wheel.

Three things, each through a public PyTorch interface:

- CPU kernels (``TORCH_LIBRARY_IMPL``): ``bmm`` threaded over the batch (the
  wheel multiplies one slice at a time without MKL), ``mm``/``addmm`` with a
  transposed right operand (``nn.Linear``'s forward) through a fast transpose
  and the plain GEMM, and SVE ``gelu``,
  ``gelu_backward`` and ``tanh`` from Arm Optimized Routines (the wheel's run
  near scalar speed on A64FX). The SVE kernels load only on CPUs with SVE.
- Settings: oneDNN off (its ARM Compute Library path is ~3x slower in
  backward on A64FX).
- Gradients over MPI: a DDP communication hook (``register_comm_hook``) that
  sums each gradient bucket with MPI_Allreduce. On Fugaku that is Fujitsu MPI
  over Tofu, bound in at run time by ``pf-run``; the process group itself
  stays gloo.
- A ``torch.distributed`` backend over MPI, ``"fjmpi"``
  (``Backend.register_backend``), for frameworks that call collectives
  directly (DeepSpeed's ZeRO): ``init_process_group("fjmpi")``.

``enable()`` applies the first two and registers the backend; ``ddp(model)`` wraps a model in DDP with
the MPI hook. In the pytorch-fugaku image both happen on ``import torch``
(``TORCH_FUGAKU=1``), so unmodified scripts get them; see ``_autoload``.
"""
import atexit
import ctypes
import os

__all__ = ["enable", "register_backend", "ddp", "mpi_comm_hook", "has_sve", "status"]

_LIB = os.path.join(os.path.dirname(__file__), "lib")
_state = {"enabled": False, "kernels": [], "mpi": None, "world": 0, "backend": False}


def has_sve():
    """True if the CPU advertises SVE (the SVE kernels would crash without it)."""
    try:
        with open("/proc/cpuinfo") as f:
            return any(line.startswith("Features") and " sve" in f" {line.split(':', 1)[1]}"
                       for line in f)
    except OSError:
        return False


def enable(kernels=True, onednn=False):
    """Load the CPU kernels and apply the A64FX settings. Idempotent."""
    import torch

    if _state["enabled"]:
        return
    _state["enabled"] = True
    if not onednn:
        torch.backends.mkldnn.enabled = False
    if kernels:
        names = ["parbmm", "tmm"] + (["fastmath"] if has_sve() else [])
        for name in names:
            torch.ops.load_library(os.path.join(_LIB, f"libtorch_fugaku_{name}.so"))
            _state["kernels"].append(name)
    register_backend()


def _backend_module():
    """The backend's extension module, with libmpi loaded RTLD_GLOBAL first: Open MPI's
    PMIx plugins (Fujitsu's included) resolve libpmix's symbols globally, and Python
    imports extension modules RTLD_LOCAL."""
    if "c10d" not in _state:
        ctypes.CDLL(os.path.join(_LIB, "libtorch_fugaku_mpi.so"), mode=ctypes.RTLD_GLOBAL)
        from torch_fugaku import _c10d as mod

        atexit.register(mod.finalize)
        _state["c10d"] = mod
    return _state["c10d"]


def _create_backend(opts, pg_options):
    return _backend_module().create(opts.store, opts.group_rank, opts.group_size,
                          list(opts.global_ranks_in_group), str(opts.group_id))


def register_backend():
    """Register ``"fjmpi"`` with torch.distributed. Idempotent; MPI starts with the first group.

    Ranks must be MPI_COMM_WORLD ranks (mpiexec + pf-run); a mismatch is an error.
    """
    import torch.distributed as dist

    if _state["backend"] or not dist.is_available():
        return
    dist.Backend.register_backend("fjmpi", _create_backend, extended_api=True, devices=["cpu"])
    _state["backend"] = True


def _mpi():
    if _state["mpi"] is None:
        lib = ctypes.CDLL(os.path.join(_LIB, "libtorch_fugaku_mpi.so"), mode=ctypes.RTLD_GLOBAL)
        lib.omni_mpi_sum_f32.argtypes = [ctypes.c_void_p, ctypes.c_long]
        _state["world"] = lib.omni_mpi_init()
        if _state["world"] < 1:
            raise RuntimeError("torch_fugaku: MPI initialisation failed")
        atexit.register(lib.omni_mpi_finalize)
        _state["mpi"] = lib
    return _state["mpi"]


def mpi_comm_hook(process_group, bucket):
    """DDP communication hook: average the bucket over MPI_COMM_WORLD.

    The same float32 sum and division by world size as DDP's default
    allreduce, with MPI as the transport. MPI rank order must match the
    process group's (true when ranks are started by mpiexec and RANK is taken
    from it, as pf-run does). Register with ``model.register_comm_hook(None,
    torch_fugaku.mpi_comm_hook)`` or use ``ddp()``.
    """
    import torch

    lib = _mpi()
    t = bucket.buffer()
    if t.dtype != torch.float32 or not t.is_contiguous():
        raise RuntimeError(f"torch_fugaku MPI hook needs contiguous float32 buckets, got {t.dtype}")
    if lib.omni_mpi_sum_f32(t.data_ptr(), t.numel()) != 0:
        raise RuntimeError("torch_fugaku: MPI_Allreduce failed")
    t.div_(_state["world"])
    fut = torch.futures.Future()
    fut.set_result(t)
    return fut


def _check_world(model):
    import torch.distributed as dist

    _mpi()
    size = dist.get_world_size() if dist.is_initialized() else 1
    if _state["world"] != size:
        raise RuntimeError(f"torch_fugaku: MPI world {_state['world']} != process group size {size}; "
                           "launch the ranks with mpiexec")


def ddp(model, **kwargs):
    """``DistributedDataParallel(model, **kwargs)`` with gradients over MPI."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    wrapped = DDP(model, **kwargs)
    if not getattr(wrapped, "_torch_fugaku_hook", False):
        _check_world(wrapped)
        wrapped.register_comm_hook(None, mpi_comm_hook)
        wrapped._torch_fugaku_hook = True
    return wrapped


def _hook_every_ddp():
    """Give every DistributedDataParallel the MPI hook, for unmodified scripts."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    build = DDP.__init__

    def init(self, *args, **kwargs):
        build(self, *args, **kwargs)
        _check_world(self)
        self.register_comm_hook(None, mpi_comm_hook)
        self._torch_fugaku_hook = True

    DDP.__init__ = init


def status():
    """What is active in this process."""
    return {"kernels": list(_state["kernels"]), "mpi_world": _state["world"], "sve": has_sve(),
            "backend": "fjmpi" if _state["backend"] else None}
