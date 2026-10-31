# Toolchain and ABI decisions

## Which binaries run where

| Component | Executes on | Produces/uses |
|---|---|---|
| CMake 3.31.10, Ninja 1.11.1.4 | x86 login | Build graph |
| Host GCC/G++ 8.5.0 | x86 login | Native protoc and SLEEF generators |
| Host CPython 3.12.8 | x86 login | Core code generation and host build tools |
| Host CPython 3.11.0 | x86 login | Binding generation and wheel assembly |
| Clang/LLVM 22.1.0 cross frontend | x86 login | AArch64 target objects |
| Target CPython 3.11.5 | A64FX compute node | Installed wheel |
| Fujitsu MPI, BLAS/LAPACK and OpenMP | A64FX compute node | Numerical/communication runtime |

The Python 3.11 host and target have compatible minor-version APIs but different
architectures. Explicitly select **target** `Python_INCLUDE_DIR`,
`Python_LIBRARY` and target NumPy headers; never use host Python headers to build
the extension. Do not execute target Python to discover its configuration on x86.
The target probe verified `SOABI=cpython-311-aarch64-linux-gnu`, eight-byte
pointers, no debug ABI and suffix `.cpython-311-aarch64-linux-gnu.so`.
Its reported `Py_ENABLE_SHARED=0` does not prevent loading extension modules;
the installation also contains `libpython3.11.so`.

## Verified site paths

Load `module load LLVM/llvmorg-22.1.0`.

```text
LLVM root: /vol0004/apps/r/OSS_CN/llvm-22.1.0
C/C++:     <LLVM>/cross_clangfx/clang-comp/bin/clang{,++}
MPI:       <LLVM>/fj-mpi/cross_mpi/bin/mpiclang{,++}
MPI hdrs:  <LLVM>/fj-mpi/cross_mpi/include/mpi/fujitsu
Fujitsu:   /opt/FJSVxtclanga/tcsds-1.2.43
Sysroot:   /opt/FJSVxos/devkit/aarch64/rfs
Linker:    /opt/FJSVxos/devkit/aarch64/bin/aarch64-linux-gnu-ld
GCC libs:  <sysroot>/usr/lib/gcc/aarch64-redhat-linux/8
```

The installed Clang configuration supplies the A64FX target, sysroot, GCC 8
search paths and cross-linker. The toolchain file also sets Linux/aarch64,
`-mcpu=a64fx -msve-vector-bits=512`, `OMPI_SKIP_MPICXX=1` and
`_GLIBCXX_USE_CXX11_ABI=1`. The MPI wrapper mentioned TCS 1.2.42 in some link
flags; CMake resolved the actual libraries under 1.2.43. Inspect wrappers and
resolved ELF dependencies when reproducing; do not assume version labels agree.

`CMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER` keeps generators native. Library,
include and package lookup are target-rooted. FindMPI's link tests succeeded
but its rooted header search initially failed, so the toolchain explicitly sets
`MPI_C_HEADER_DIR` and `MPI_CXX_HEADER_DIR` to the real wrapper header directory.
It does not fake a successful MPI compile/link test.

## Fujitsu math and OpenMP

The tested Clang-compatible libraries are **`fjlapacksve`** (serial) and
**`fjlapackexsve`** (threaded). These are distinct from the traditional
`libssl2*.a` interface. Do not substitute an SSL2 archive merely because
FindBLAS detects it: the naming/ABI assumptions differ. The LP64 Fortran entry
points used here have trailing underscores, such as `dgemm_` and `dgesv_`.
Fujitsu describes this cross-frontend math-library use as experimental; the
validation here establishes observed behavior, not vendor support.

The complete threaded link set, preserving runtime precedence, is:

```sh
-L<fj>/lib64 -lfjlapackexsve -lfjomphk -lfjomp \
  -lfj90i -lfj90f -lfjsrcinfo -lfjcrt -lfjompcrt -lelf
```

`libelf` is required by the Fujitsu Fortran runtime. Without it the LAPACK
probe fails with unresolved `elf_*` symbols. The custom `FindLAPACK.cmake`
checks a real `dgesv_` link with this complete library set.

Fujitsu `libfj90i` exports OpenMP stub APIs that can shadow the active runtime.
Serial-BLAS experiments worked after preloading LLVM `libomp`, but mixing that
runtime with **threaded** Fujitsu BLAS segfaulted. The final solution uses
Fujitsu `libfjomp`, which implements the LLVM-generated `__kmpc_*` calls.

For CMake, lower OpenMP through `-Xclang -fopenmp -I<workspace>/omp-include`
and link imported `OpenMP::OpenMP_CXX` to `libfjomp.so`. `omp-include/omp.h`
is a symlink to the Fujitsu header, avoiding a broad Fujitsu include directory
overriding standard headers. Plain driver `-fopenmp` at the link step would
automatically add LLVM `libomp`. The raw probe alternatively compiled with
`-fopenmp -c`, then linked without `-fopenmp`.

Reject direct `libomp`, `libgomp` or `libiomp` dependencies in the final core,
binding and extension libraries. Verify active threads on compute nodes too;
an ELF dependency alone did not catch the stub-shadowing problem.

## CPU feature scope

This is an A64FX build with 512-bit SVE compiler targeting and SLEEF built for
that vector length. A64FX lacks BF16 instructions. Force the inappropriate
`CXX_ARM_BF16_FOUND` and stock `CXX_SVE256_FOUND` variants off. Keep
`USE_NATIVE_ARCH=OFF`: the build host is x86. This recipe does **not** apply the
older port's custom PyTorch SVE512 dispatch/kernel patches.

CUDA, ROCm, XPU, MPS, MAGMA, oneDNN/MKLDNN, FBGEMM, KleidiAI, XNNPACK,
NNPACK, QNNPACK, ITT, Kineto runtime and NUMA support are disabled in this
baseline. Gloo and TensorPipe are disabled. CPU operator coverage, quantization,
JIT/compile features and real model performance need their own validation;
passing GEMM and MPI smoke tests does not establish all of them.
