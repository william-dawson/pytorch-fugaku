#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_DYNAMIC=FALSE
case "${1:?}" in
  smoke|async|benchmark) script="python-$1.py" ;;
  largepage) exec "$root/bin/python3.11-fugaku" -u "$root/validate-new-runtime.py" ;;
  *) exit 2 ;;
esac
exec "$root/bin/python3.11-fugaku" -u "$root/test-fresh-python.py" "$script"
