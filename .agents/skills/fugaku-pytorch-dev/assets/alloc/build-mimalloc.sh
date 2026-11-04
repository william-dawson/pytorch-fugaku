#!/bin/bash
# Cross-build mimalloc as a shared library for LD_PRELOAD into the LLVM build's python on A64FX.
# Run on an x86 login node, in a login shell (`module` must exist: `bash -lc 'bash build-mimalloc.sh'`).
#
#   WORK      the cross-build workspace: has fugaku-llvm.cmake, host-venv/ (cmake, ninja) and
#             mimalloc-source.tar.gz (PyTorch's pinned third_party/mimalloc, 2.2.6, MIT)
#   OUT       where to build (default: $WORK/h2h-alloc)
#
# Tested 2026-11-03: produced libmimalloc.so.2.2 exporting malloc/free/calloc/realloc/
# posix_memalign/aligned_alloc. See references/performance.md for how it is used and why.
set -euo pipefail
WORK=${WORK:?set WORK to the cross-build workspace}
OUT=${OUT:-$WORK/h2h-alloc}
mkdir -p "$OUT/src"
cd "$OUT"
tar xzf "$WORK/mimalloc-source.tar.gz" -C src
module load LLVM/llvmorg-22.1.0
export PATH="$WORK/host-venv/bin:$PATH"
cmake -S src/third_party/mimalloc -B build -G Ninja --toolchain "$WORK/fugaku-llvm.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DMI_BUILD_SHARED=ON -DMI_BUILD_STATIC=OFF -DMI_BUILD_OBJECT=OFF \
  -DMI_BUILD_TESTS=OFF -DMI_OVERRIDE=ON
cmake --build build --parallel 4
file build/libmimalloc.so.2.2
