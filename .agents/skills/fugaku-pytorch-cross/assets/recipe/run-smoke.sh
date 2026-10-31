#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
unset LD_PRELOAD
export LD_LIBRARY_PATH="$PWD/target/lib:/opt/FJSVxtclanga/tcsds-1.2.43/lib64"
export OMP_NUM_THREADS=48 OMP_DYNAMIC=FALSE
mpiexec -n 2 ./build-smoke/torch-mpi-smoke
printf 'OVERALL PASS\n'
