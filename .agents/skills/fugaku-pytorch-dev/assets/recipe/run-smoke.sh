#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE
LD_LIBRARY_PATH=/opt/FJSVxtclanga/tcsds-1.2.43/lib64 mpiexec -n 2 /usr/bin/env -u LD_LIBRARY_PATH ./build-smoke/torch-mpi-smoke
printf 'OVERALL PASS\n'
