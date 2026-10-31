#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
trap 'code=$?; printf "%s\n" "$code" > exit-code.txt; if [[ "$code" != 0 ]]; then printf "failed\n" > status.txt; fi' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export MAX_JOBS=${MAX_JOBS:-8}
[[ "$MAX_JOBS" =~ ^[1-8]$ ]] || { echo 'MAX_JOBS must be 1..8'; exit 2; }
export PYTHONNOUSERSITE=1 PYTORCH_BUILD_VERSION=2.11.0+fugaku.llvm
export PYTORCH_BLAS_F2C=OFF PYTORCH_BLAS_USE_CBLAS_DOT=OFF
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CMAKE_TOOLCHAIN_FILE LD_PRELOAD
printf 'host-tools\n' > status.txt
host_python=/vol0402/data/ra000009/u10035/fminiconda3/bin/python3.12
"$host_python" -c 'import platform,sys; assert platform.machine()=="x86_64"; assert sys.version_info[:2]==(3,12); print(sys.version)'
[[ -x host-venv/bin/python ]] || "$host_python" -m venv "$root/host-venv"
host-venv/bin/python -m pip install --no-index --no-cache-dir --find-links "$root/wheels" -r host-requirements.txt
module load LLVM/llvmorg-22.1.0
export LD=/opt/FJSVxos/devkit/aarch64/bin/aarch64-linux-gnu-ld
export PATH="$root/host-venv/bin:$PATH"
cmake --version
ninja --version
clang++ --version
mkdir -p omp-include target
ln -sfn /opt/FJSVxtclanga/tcsds-1.2.43/include/omp.h omp-include/omp.h
src="$root/src/pytorch"
[[ -f "$src/third_party/eigen/Eigen/Core" ]] || { echo 'Missing Eigen'; exit 2; }
# Apply the reference recipe's static-SLEEF import/include fix idempotently.
if git -C "$src" apply --check "$root/aten-sleef.patch"; then
  git -C "$src" apply "$root/aten-sleef.patch"
else
  git -C "$src" apply --reverse --check "$root/aten-sleef.patch"
fi
if git -C "$src" apply --reverse --check "$root/generated-paths.patch" 2>/dev/null; then
  :
else
  git -C "$src" apply --check "$root/generated-paths.patch"
  git -C "$src" apply "$root/generated-paths.patch"
fi
printf 'BUILD_HOST=%s MAX_JOBS=%s\n' "$(hostname)" "$MAX_JOBS"
if git -C "$src" apply --reverse --check "$root/semaphore-version.patch" 2>/dev/null; then
  :
else
  git -C "$src" apply --check "$root/semaphore-version.patch"
  git -C "$src" apply "$root/semaphore-version.patch"
fi

if git -C "$src" apply --reverse --check "$root/atomic-value-type.patch" 2>/dev/null; then
  :
else
  git -C "$src" apply --check "$root/atomic-value-type.patch"
  git -C "$src" apply "$root/atomic-value-type.patch"
fi

if git -C "$src" apply --reverse --check "$root/filesystem-link.patch" 2>/dev/null; then
  :
else
  git -C "$src" apply --check "$root/filesystem-link.patch"
  git -C "$src" apply "$root/filesystem-link.patch"
fi

printf 'host-protoc\n' > status.txt
cmake -S "$src/third_party/protobuf/cmake" -B build-protoc-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_WITH_ZLIB=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX="$root/host-tools"
cmake --build build-protoc-host --parallel "$MAX_JOBS"
cmake --install build-protoc-host
file host-tools/bin/protoc
host-tools/bin/protoc --version

sleef_flags=(-DSLEEF_BUILD_SHARED_LIBS=OFF -DSLEEF_BUILD_DFT=OFF -DSLEEF_BUILD_GNUABI_LIBS=OFF -DSLEEF_BUILD_TESTS=OFF -DSLEEF_BUILD_SCALAR_LIB=OFF -DSLEEF_BUILD_QUAD=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
printf 'host-sleef-generators\n' > status.txt
cmake -S "$src/third_party/sleef" -B build-sleef-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc "${sleef_flags[@]}"
cmake --build build-sleef-host --parallel "$MAX_JOBS"
printf 'target-sleef\n' > status.txt
cmake -S "$src/third_party/sleef" -B build-sleef-target -G Ninja \
  --toolchain "$root/fugaku-llvm.cmake" -DCMAKE_BUILD_TYPE=Release "${sleef_flags[@]}" \
  -DNATIVE_BUILD_DIR="$root/build-sleef-host" -DSVE_VECTOR_BITS=512 \
  -DCMAKE_INSTALL_PREFIX="$root/target"
cmake --build build-sleef-target --parallel "$MAX_JOBS"
cmake --install build-sleef-target

printf 'configure-pytorch-mpi\n' > status.txt
fj=/opt/FJSVxtclanga/tcsds-1.2.43/lib64
blas="$fj/libfjlapackexsve.so;$fj/libfjomphk.so;$fj/libfjomp.so;$fj/libfj90i.so;$fj/libfj90f.so;$fj/libfjsrcinfo.so;$fj/libfjcrt.so;$fj/libfjompcrt.so;elf"
cmake -S "$src" -B build-pytorch -G Ninja --toolchain "$root/fugaku-llvm.cmake" \
  -UFUGAKU_LAPACK_LINKS \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/target" \
  -DCMAKE_MODULE_PATH="$root/cmake" -DCMAKE_PREFIX_PATH="$root/target" \
  -DPython_EXECUTABLE="$root/host-venv/bin/python" \
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE="$root/host-tools/bin/protoc" \
  -DPROTOBUF_PROTOC_EXECUTABLE="$root/host-tools/bin/protoc" -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
  -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF -DBUILD_TESTING=OFF -DBUILD_BINARY=OFF \
  -DUSE_DISTRIBUTED=ON -DUSE_MPI=ON -DUSE_C10D_MPI=ON -DUSE_GLOO=OFF -DUSE_TENSORPIPE=OFF \
  -DUSE_OPENMP=ON -DBLAS=Generic "-DBLAS_LIBRARIES=$blas" \
  -DUSE_SYSTEM_SLEEF=ON -DSLEEF_LIBRARY="$root/target/lib/libsleef.a" -DSLEEF_INCLUDE="$root/target/include" \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF -DUSE_MPS=OFF -DUSE_MAGMA=OFF \
  -DUSE_MKLDNN=OFF -DUSE_FBGEMM=OFF -DUSE_KLEIDIAI=OFF -DUSE_XNNPACK=OFF \
  -DUSE_NNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_ITT=OFF -DUSE_KINETO=OFF \
  -DUSE_NUMA=OFF -DUSE_NUMPY=OFF -DUSE_NATIVE_ARCH=OFF -DCXX_ARM_BF16_FOUND=OFF -DCXX_SVE256_FOUND=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
host-venv/bin/python check-build.py configuration
printf 'compile-pytorch-mpi\n' > status.txt
cmake --build build-pytorch --parallel "$MAX_JOBS"
printf 'install-and-check\n' > status.txt
cmake --install build-pytorch
host-venv/bin/python check-build.py artifacts
printf 'build-smoke-test\n' > status.txt
cmake -S smoke -B build-smoke -G Ninja --toolchain "$root/fugaku-llvm.cmake" \
  -DCMAKE_PREFIX_PATH="$root/target" -DCMAKE_BUILD_TYPE=Release
cmake --build build-smoke --parallel "$MAX_JOBS"
printf 'ready-for-compute-node-validation\n' > status.txt
