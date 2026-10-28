#!/bin/bash
# Build torch_fugaku's libraries against the installed torch and Open MPI and
# install the package into site-packages. Run inside the image build.
set -euo pipefail
cd "$(dirname "$0")"
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
TL=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
ABI=$(python -c "import torch; print(int(torch.compiled_with_cxx11_abi()))")
OUT=torch_fugaku/lib
mkdir -p "$OUT"
TORCH_FLAGS=(-std=c++17 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=$ABI -I"$TL/include"
  -I"$TL/include/torch/csrc/api/include" -L"$TL/lib" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TL/lib")
# parbmm: generic aarch64. OpenMP is required: at::parallel_for is an inline
# pragma, and without -fopenmp the override silently runs serially.
g++ -O3 -fopenmp csrc/parbmm.cpp "${TORCH_FLAGS[@]}" -o "$OUT/libtorch_fugaku_parbmm.so"
# fastmath: vector-length-agnostic SVE; loaded only on CPUs with SVE.
g++ -O3 -fopenmp -march=armv8.2-a+sve -Icsrc csrc/fastmath.cpp "${TORCH_FLAGS[@]}" -o "$OUT/libtorch_fugaku_fastmath.so"
gcc -O2 -shared -fPIC csrc/mpi_allreduce.c -I/opt/openmpi/include -L/opt/openmpi/lib -lmpi \
  -o "$OUT/libtorch_fugaku_mpi.so"
# The torch.distributed backend "fjmpi": a Python extension module over c10d::Backend.
PYINC=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")
g++ -O2 csrc/c10d_mpi.cpp "${TORCH_FLAGS[@]}" -I"$PYINC" -I/opt/openmpi/include -ltorch_python \
  -L/opt/openmpi/lib -lmpi -o torch_fugaku/_c10d.so
cp -r torch_fugaku "$SITE/"
cp torch_fugaku.pth "$SITE/"
echo "torch_fugaku installed in $SITE"
