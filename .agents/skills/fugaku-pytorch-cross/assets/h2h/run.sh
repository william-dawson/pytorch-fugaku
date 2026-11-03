#!/bin/bash
# Head-to-head at scale: each case in H2H_CASES (see rank.sh) in turn on the same allocation.
# Needs #PJM --llio cn-cache-size=384Mi (h2h-mlock-test: the 128 MiB default refaults torch code).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
suite="results.${PJM_JOBID}"
mkdir -p "$suite"
trap 'code=$?; printf "%s\n" "$code" > "$here/$suite/exit.txt"' EXIT
export PF_HOSTLIB="$here/.pf-hostlib.${PJM_JOBID}"
# H2H_RPN ranks per node (default 4, one per CMG; benchkit also uses 2 x 24 threads).
bash "$here/../h2h-paired/pf/pf-setup" "${H2H_RPN:-4}"
set -a; source "$PF_HOSTLIB/pf.env"; set +a
export H2H_MODE=resources
{ env | grep -E '^(PJM_LLIO_CN_CACHE_SIZE|PJM_NODE|WORLD_SIZE|H2H_)' | sort; } > "$suite/env.txt"
failed=0
for case in ${H2H_CASES:?}; do
  export H2H_OUTPUT_REL="$suite/$case" H2H_OUTPUT="$here/$suite/$case"
  mkdir -p "$H2H_OUTPUT"
  printf '%s start %s\n' "$(date -u +%FT%TZ)" "$case" >> "$suite/progress.txt"
  if timeout "${H2H_CASE_TIMEOUT:-1500}" mpiexec -n "$WORLD_SIZE" /bin/bash "$here/rank.sh" "$case"; then
    printf '0\n' > "$H2H_OUTPUT/exit.txt"
  else
    code=$?
    printf '%s\n' "$code" > "$H2H_OUTPUT/exit.txt"
    failed=1
  fi
  printf '%s end %s\n' "$(date -u +%FT%TZ)" "$case" >> "$suite/progress.txt"
done
exit "$failed"
