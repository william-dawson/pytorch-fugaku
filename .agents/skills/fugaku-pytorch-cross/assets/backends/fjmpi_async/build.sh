#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
cd "$here"
if [[ "${FJ_PORT_LOCK_INHERITED:-0}" != 1 ]]; then
  exec 9>build.lock
  flock -n 9 || { echo 'Async build already locked'; exit 1; }
fi
trap 'code=$?; printf "%s\n" "$code" > build-exit.txt' EXIT
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_PRELOAD PYTHONPATH
module load LLVM/llvmorg-22.1.0
export PATH="$root/host-venv/bin:$PATH"
cmake -S "$here" -B "$here/build" -G Ninja --toolchain "$root/fugaku-llvm.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$root/target" -DCMAKE_MODULE_PATH="$root/cmake" \
  -DPYCROSS_ROOT="$root" -DPython_EXECUTABLE="$root/host-venv311/bin/python" \
  -DPython_INCLUDE_DIR=/vol0402/data/ra000009/u10035/miniconda3/include/python3.11 \
  -DPython_LIBRARY=/vol0402/data/ra000009/u10035/miniconda3/lib/libpython3.11.so \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$here/build" --parallel 1
file "$here/fjmpi_async/_c10d.cpython-311-aarch64-linux-gnu.so"
