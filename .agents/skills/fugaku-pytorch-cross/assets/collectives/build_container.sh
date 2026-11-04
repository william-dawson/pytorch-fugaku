#!/bin/bash
# Build torch_fugaku's fjmpi backend (repo c10d_mpi.cpp with TORCH_FUGAKU_RS_ALGO) inside the
# pytorch-fugaku image, as torch_fugaku/build.sh does, into h2h-rsalgo/container/torch_fugaku
# (put first on PYTHONPATH; the image's other libraries are copied from its own install).
set -euo pipefail
W=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross
singularity exec -B $W:/cross $W/pytorch-fugaku-b5879f2.sif bash -c '
  set -euo pipefail
  SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  rm -rf /cross/h2h-rsalgo/container && mkdir -p /cross/h2h-rsalgo/container
  cp -r "$SITE/torch_fugaku" /cross/h2h-rsalgo/container/
  TL=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
  ABI=$(python -c "import torch; print(int(torch.compiled_with_cxx11_abi()))")
  PYINC=$(python -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")
  g++ -O2 /cross/h2h-rsalgo/c10d_mpi.cpp -std=c++17 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -I"$TL/include" -I"$TL/include/torch/csrc/api/include" -L"$TL/lib" -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,"$TL/lib" -I"$PYINC" -I/opt/openmpi/include -ltorch_python -L/opt/openmpi/lib -lmpi \
    -o /cross/h2h-rsalgo/container/torch_fugaku/_c10d.so
  ls -la /cross/h2h-rsalgo/container/torch_fugaku/'
echo BUILT
