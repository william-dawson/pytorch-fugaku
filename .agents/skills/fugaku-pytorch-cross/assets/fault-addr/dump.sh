#!/bin/bash
# Re-read job 53270101's major-fault perf capture (recorded with -d) and print each sample's
# faulting data address next to its instruction pointer, for classification against /proc maps.
set -uo pipefail
src=/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross/h2h-fault-attribution/results.53270101/native_resources
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out="$here/results.${PJM_JOBID}"
mkdir -p "$out"
which perf > "$out/perf-version.txt" 2>&1; perf --version >> "$out/perf-version.txt" 2>&1
for r in 0 4; do
  perf report -i "$src/observer.rank$r.perf.data" --header-only > "$out/header.rank$r.txt" 2>&1
  perf script -i "$src/observer.rank$r.perf.data" -G -F tid,time,addr,ip,sym,dso \
    > "$out/addr.rank$r.txt" 2> "$out/addr.rank$r.err"
  echo "rank$r exit $?" >> "$out/status.txt"
done
