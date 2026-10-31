#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
trap 'code=$?; printf "%s\n" "$code" > python-exit-code.txt; if [[ "$code" != 0 ]]; then printf "failed\n" > python-status.txt; fi' EXIT
export MAX_JOBS=${MAX_JOBS:-8}
[[ "$MAX_JOBS" =~ ^[1-8]$ ]] || exit 2
export PYTHONNOUSERSITE=1 PYTORCH_BUILD_VERSION=2.11.0+fugaku.llvm
export PYTORCH_BUILD_NUMBER=1
export PYTORCH_BLAS_F2C=OFF PYTORCH_BLAS_USE_CBLAS_DOT=OFF
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_PRELOAD PYTHONPATH
printf 'host-python-311\n' > python-status.txt
host=/vol0402/data/ra000009/u10035/fminiconda3/envs/front/bin/python3.11
target=/vol0402/data/ra000009/u10035/miniconda3
"$host" -c 'import platform,sys; assert platform.machine()=="x86_64"; assert sys.version_info[:2]==(3,11); import yaml; print(sys.version)'
[[ -x host-venv311/bin/python ]] || "$host" -m venv --system-site-packages host-venv311
host-venv311/bin/python -m pip install --no-index --find-links wheels --no-deps typing-extensions==4.15.0 packaging==25.0 setuptools==80.9.0 wheel==0.45.1
module load LLVM/llvmorg-22.1.0
export PATH="$root/host-venv/bin:$PATH"
export LD=/opt/FJSVxos/devkit/aarch64/bin/aarch64-linux-gnu-ld
export LIBTORCH_LIB_PATH="$root/target/lib"
python="$root/host-venv311/bin/python"
"$python" python-initial-cache.py
printf 'BUILD_HOST=%s MAX_JOBS=%s\n' "$(hostname)" "$MAX_JOBS"
printf 'configure-python\n' > python-status.txt
python_args=("-DPython_EXECUTABLE=$python" "-DPython_INCLUDE_DIR=$target/include/python3.11" "-DPython_LIBRARY=$target/lib/libpython3.11.so")
cmake -S src/pytorch -B build-python -G Ninja --toolchain "$root/fugaku-llvm.cmake" \
  -C python-initial-cache.cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$root/python-prefix" -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,/opt/FJSVxtclanga/tcsds-1.2.43/lib64 -Wl,-rpath-link,$root/target/lib -Wl,-rpath-link,/opt/FJSVxtclanga/tcsds-1.2.43/lib64" \
  -DCMAKE_PREFIX_PATH="$root/target" -DCMAKE_MODULE_PATH="$root/cmake" \
  -DBUILD_LIBTORCHLESS=ON -DBUILD_PYTHON=ON -DBUILD_TEST=OFF -DBUILD_TESTING=OFF -DBUILD_BINARY=OFF \
  -DUSE_NUMPY=ON "-DPython_NumPy_INCLUDE_DIR=$root/runtime-site/numpy/_core/include" \
  -DUSE_DISTRIBUTED=ON -DUSE_MPI=ON -DUSE_C10D_MPI=ON \
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE="$root/host-tools/bin/protoc" \
  -DPROTOBUF_PROTOC_EXECUTABLE="$root/host-tools/bin/protoc" -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${python_args[@]}"
"$python" check-python-build.py configuration
printf 'compile-python\n' > python-status.txt
cmake --build build-python --parallel "$MAX_JOBS"
cmake --install build-python
printf 'build-python-extension\n' > python-status.txt
cmake -S python-extension -B build-python-extension -G Ninja --toolchain "$root/fugaku-llvm.cmake" \
  -DCMAKE_BUILD_TYPE=Release "-DPYCROSS_ROOT=$root" "${python_args[@]}"
cmake --build build-python-extension --parallel "$MAX_JOBS"
printf 'package-wheel\n' > python-status.txt
"$python" package-python.py
"$python" check-python-build.py artifacts
"$python" -m pip install --no-index --no-deps --no-compile --upgrade --target "$root/wheel-install" \
  --platform linux_aarch64 --python-version 3.11 --implementation cp --abi cp311 "$root"/dist/torch-2.11.0+fugaku.llvm-cp311-cp311-linux_aarch64.whl
printf 'ready-for-python-runtime-validation\n' > python-status.txt
