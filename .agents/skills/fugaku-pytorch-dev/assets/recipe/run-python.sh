#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode=${1:?}
ranks=${2:?}
case "$mode" in
  smoke) [[ "$ranks" == 2 ]]; export OMP_NUM_THREADS=12 ;;
  benchmark) [[ "$ranks" =~ ^[124]$ ]]; export OMP_NUM_THREADS=$((48/ranks)) ;;
  async|largepage) [[ "$ranks" == 2 ]]; export OMP_NUM_THREADS=12 ;;
  *) exit 2 ;;
esac
LD_LIBRARY_PATH=/opt/FJSVxtclanga/tcsds-1.2.43/lib64 mpiexec -n "$ranks" /bin/bash ./run-python-rank.sh "$mode"
printf 'OVERALL PASS\n'
