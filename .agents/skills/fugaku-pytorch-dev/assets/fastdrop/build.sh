#!/bin/bash
# Build libfastdrop.so against the LLVM wheel. Run in a login shell on the x86 login node:
#   bash -lc 'bash build.sh'      (expects this directory at $WORK/<anything>/, WORK = ../..)
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/.." && pwd)
cd "$here"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_PRELOAD PYTHONPATH
module load LLVM/llvmorg-22.1.0
export PATH="$root/host-venv/bin:$PATH"
cmake -S "$here" -B "$here/build" -G Ninja --toolchain "$root/fugaku-llvm.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$root/target" -DCMAKE_MODULE_PATH="$root/cmake" \
  -DPYCROSS_ROOT="$root" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$here/build" --parallel 2
file "$here/libfastdrop.so"
