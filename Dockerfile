# PyTorch for Fugaku (A64FX): the official aarch64 PyTorch wheel plus
# torch_fugaku (A64FX CPU kernels and gradients over MPI, through public
# PyTorch interfaces). Built on any arm64 Linux host; nothing from Fujitsu is
# in the image: fugaku/pf-run binds Fujitsu MPI in at run time.
FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370

ARG TORCH_VERSION=2.11.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Open MPI 4.0.1, Fujitsu MPI's base, without extensions: torch_fugaku's MPI
# library is compiled against it, and on Fugaku the same binary runs on
# Fujitsu MPI. Off Fugaku the image uses this Open MPI itself.
RUN curl -sSL https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2 | tar xj -C /tmp && \
    cd /tmp/openmpi-4.0.1 && \
    ./configure --prefix=/opt/openmpi --disable-mpi-fortran --disable-static \
      --enable-mpi-ext=none CFLAGS="-O2 -fcommon" > /dev/null && \
    make -j"$(nproc)" install > /dev/null && \
    cd / && rm -rf /tmp/openmpi-4.0.1

RUN pip install --no-cache-dir torch==${TORCH_VERSION}

COPY torch_fugaku /tmp/torch_fugaku
RUN /tmp/torch_fugaku/build.sh && rm -rf /tmp/torch_fugaku

# Run-time configuration. On Fugaku, pf-run binds /opt/FJSVxtclanga and the
# compute node's interconnect libraries (into /opt/pytorch-fugaku/hostlib);
# then libmpi.so.40 resolves to Fujitsu MPI. Elsewhere those paths are absent
# and Open MPI is used. fjomp/ serves PF_FJBLAS=1 only (README).
RUN mkdir -p /opt/pytorch-fugaku/fujitsu-mpi /opt/pytorch-fugaku/hostlib /opt/pytorch-fugaku/fjomp && \
    ln -s /opt/FJSVxtclanga/tcsds-latest/lib64/libmpi.so.0 /opt/pytorch-fugaku/fujitsu-mpi/libmpi.so.40 && \
    ln -s /opt/FJSVxtclanga/tcsds-latest/lib64/libfjomphk.so /opt/pytorch-fugaku/fjomp/libgomp.so.1
ENV PATH=/opt/openmpi/bin:$PATH \
    LD_LIBRARY_PATH=/opt/pytorch-fugaku/fujitsu-mpi:/opt/pytorch-fugaku/hostlib:/opt/FJSVxtclanga/tcsds-latest/lib64:/opt/openmpi/lib \
    TORCH_FUGAKU=1 \
    OMP_WAIT_POLICY=ACTIVE \
    PYTHONNOUSERSITE=1
COPY tests /opt/pytorch-fugaku/tests
COPY tools /opt/pytorch-fugaku/tools
