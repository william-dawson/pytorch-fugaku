#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unset LD_PRELOAD PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root/wheel-install:$root/runtime-site"
export LD_LIBRARY_PATH="$root/wheel-install/torch/lib:/opt/FJSVxtclanga/tcsds-1.2.43/lib64"
export OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
case "${1:?}" in
  smoke) script=python-smoke.py ;;
  benchmark) script=python-benchmark.py ;;
  async) script=python-async.py ;;
  *) exit 2 ;;
esac
exec /vol0402/data/ra000009/u10035/miniconda3/bin/python3.11 -u "$root/$script"
