"""fjmpi_port (the container's pinned MPI backend, rebuilt against the native LLVM torch wheel)
plus allgathers that overlap with computation; see c10d_mpi_async.cpp."""
import atexit
import ctypes

_registered = False
_module = None

def _load():
    global _module
    if _module is None:
        ctypes.CDLL('/opt/FJSVxtclanga/tcsds-1.2.43/lib64/libmpi.so',mode=ctypes.RTLD_GLOBAL)
        from . import _c10d
        _module = _c10d
        atexit.register(_module.finalize)
    return _module

def _create(opts, pg_options):
    return _load().create(opts.store,opts.group_rank,opts.group_size,
                          list(opts.global_ranks_in_group),str(opts.group_id))

def register_backend():
    global _registered
    if not _registered:
        import torch.distributed as dist
        dist.Backend.register_backend('fjmpi_async',_create,extended_api=True,devices=['cpu'])
        _registered=True

def stats():
    """Seconds blocked in the backend by kind, and gather counts, since the last reset_stats()."""
    return _load().stats()

def reset_stats():
    _load().reset_stats()

def trace():
    """(t, kind, id, bytes) events since the last call, if TORCH_FUGAKU_TRACE=1; clears them."""
    return _load().trace()
