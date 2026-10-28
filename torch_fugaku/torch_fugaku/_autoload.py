"""Apply torch_fugaku when torch is first imported, so unmodified scripts get it.

Imported from ``torch_fugaku.pth`` at interpreter start; does nothing until
``import torch`` finishes, then, with ``TORCH_FUGAKU=1`` (the image default):

- ``torch_fugaku.enable()``                    unless TORCH_FUGAKU_KERNELS=0
  (oneDNN stays on with TORCH_FUGAKU_ONEDNN=1)
- the MPI comm hook on every DDP model          TORCH_FUGAKU_MPI=auto|1|0
  ("auto": when an MPI launcher started the process, i.e. PMIX_RANK or
  OMPI_COMM_WORLD_RANK is set, and WORLD_SIZE > 1)
"""
import importlib.abc
import importlib.machinery
import os
import sys


def _apply(torch):
    import torch_fugaku

    torch_fugaku.enable(kernels=os.environ.get("TORCH_FUGAKU_KERNELS", "1") != "0",
                        onednn=os.environ.get("TORCH_FUGAKU_ONEDNN", "0") == "1")
    mpi = os.environ.get("TORCH_FUGAKU_MPI", "auto")
    launched = any(v in os.environ for v in ("PMIX_RANK", "OMPI_COMM_WORLD_RANK"))
    if mpi == "1" or (mpi == "auto" and launched and int(os.environ.get("WORLD_SIZE", "1")) > 1):
        torch_fugaku._hook_every_ddp()


class _AfterTorchImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _apply(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("TORCH_FUGAKU", "0") == "1":
    sys.meta_path.insert(0, _AfterTorchImport())
