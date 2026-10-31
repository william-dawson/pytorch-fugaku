#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trap 'code=$?; printf "%s\n" "$code" > "validation-exit.${PJM_JOBID}.txt"' EXIT
mode=${1:?}
ranks=${2:?}
case "$mode" in
  libtorch) [[ "$ranks" == 2 ]]; bash ./run-smoke.sh ;;
  smoke|async|benchmark) bash ./run-python.sh "$mode" "$ranks" ;;
  *) exit 2 ;;
esac
