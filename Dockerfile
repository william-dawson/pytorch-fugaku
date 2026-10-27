# PyTorch for Fugaku (A64FX), built from source on any arm64 Linux host.
#
# torch 2.11.0 with the patches in patches/ (512-bit SVE kernels, batch-parallel
# bmm, Arm Optimized Routines erf/tanh, staged MPI allreduce), NVPL BLAS as
# PyTorch's own aarch64 wheels, gloo, and the MPI backend built against Open MPI
# 4.0.1: Fujitsu MPI's base, so on Fugaku the same binary runs on Fujitsu MPI.
# Nothing from Fujitsu is in the image; fugaku/pf-run binds it in at run time.
FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370

ARG PYTORCH_REF=v2.11.0
ARG MAX_JOBS=
ARG NVPL_BLAS=0.3.0
ARG NVPL_LAPACK=0.2.3.1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates bzip2 xz-utils && \
    rm -rf /var/lib/apt/lists/*

# Open MPI 4.0.1 without extensions: with its CUDA extension, the PyTorch MPI
# backend would call MPIX_Query_cuda_support, which Fujitsu MPI does not have.
RUN curl -sSL https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2 | tar xj -C /tmp && \
    cd /tmp/openmpi-4.0.1 && \
    ./configure --prefix=/opt/openmpi --disable-mpi-fortran --disable-static \
      --enable-mpi-ext=none CFLAGS="-O2 -fcommon" > /dev/null && \
    make -j"$(nproc)" install > /dev/null && \
    cd / && rm -rf /tmp/openmpi-4.0.1

# NVPL, from NVIDIA's redistributable archives (as PyTorch's install_nvpl.sh).
RUN mkdir -p /opt/nvpl && cd /tmp && \
    for t in nvpl_blas/linux-sbsa/nvpl_blas-linux-sbsa-${NVPL_BLAS}-archive \
             nvpl_lapack/linux-sbsa/nvpl_lapack-linux-sbsa-${NVPL_LAPACK}-archive; do \
      curl -sSL "https://developer.download.nvidia.com/compute/nvpl/redist/$t.tar.xz" | tar xJ && \
      cp -r "$(basename "$t")"/lib "$(basename "$t")"/include /opt/nvpl/ && rm -rf "$(basename "$t")"; \
    done

COPY patches /opt/pytorch-fugaku/patches
ENV PATH=/opt/openmpi/bin:$PATH
RUN git clone --branch ${PYTORCH_REF} --depth 1 --recurse-submodules --shallow-submodules -j8 \
      https://github.com/pytorch/pytorch.git /tmp/pytorch && \
    cd /tmp/pytorch && \
    for p in /opt/pytorch-fugaku/patches/*.patch; do git apply "$p"; done && \
    pip install --no-cache-dir -r requirements.txt && \
    export BLAS=NVPL CMAKE_PREFIX_PATH=/opt/nvpl MPI_HOME=/opt/openmpi \
      LD_LIBRARY_PATH=/opt/nvpl/lib:/opt/openmpi/lib \
      USE_CUDA=0 USE_ROCM=0 USE_XPU=0 USE_MKLDNN=0 USE_FBGEMM=0 USE_KLEIDIAI=0 \
      CMAKE_ARGS=-DXNNPACK_ENABLE_KLEIDIAI=OFF \
      USE_DISTRIBUTED=1 USE_GLOO=1 USE_MPI=1 USE_OPENMP=1 BUILD_TEST=0 \
      CMAKE_BUILD_TYPE=Release PYTORCH_BUILD_VERSION=${PYTORCH_REF#v}+fugaku PYTORCH_BUILD_NUMBER=1 \
      MAX_JOBS=${MAX_JOBS:-$(nproc)} && \
    python -m pip wheel . --no-build-isolation --no-deps -w /opt/pytorch-fugaku/dist && \
    pip install --no-cache-dir /opt/pytorch-fugaku/dist/torch-*.whl && \
    cd / && rm -rf /tmp/pytorch /root/.cache

# Run-time configuration. On Fugaku, fugaku/pf-run binds /opt/FJSVxtclanga and
# the compute node's interconnect libraries (into /opt/pytorch-fugaku/hostlib);
# then libmpi.so.40 below resolves to Fujitsu MPI. Elsewhere those paths are
# absent and the image's own Open MPI is used.
# fjomp/ is used only with PF_FJBLAS=1: torch's libgomp.so.1 becomes Fujitsu's
# libfjomphk, which serves libgomp's interface on Fujitsu's OpenMP runtime.
RUN mkdir -p /opt/pytorch-fugaku/fujitsu-mpi /opt/pytorch-fugaku/hostlib /opt/pytorch-fugaku/fjomp && \
    ln -s /opt/FJSVxtclanga/tcsds-latest/lib64/libmpi.so.0 /opt/pytorch-fugaku/fujitsu-mpi/libmpi.so.40 && \
    ln -s /opt/FJSVxtclanga/tcsds-latest/lib64/libfjomphk.so /opt/pytorch-fugaku/fjomp/libgomp.so.1
ENV LD_LIBRARY_PATH=/opt/pytorch-fugaku/fujitsu-mpi:/opt/pytorch-fugaku/hostlib:/opt/FJSVxtclanga/tcsds-latest/lib64:/opt/nvpl/lib:/opt/openmpi/lib \
    OMP_WAIT_POLICY=ACTIVE \
    TORCH_MPI_STAGE_BYTES=8388608 \
    PYTHONNOUSERSITE=1
COPY tools /opt/pytorch-fugaku/tools
