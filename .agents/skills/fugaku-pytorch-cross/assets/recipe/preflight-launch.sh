#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
exec 9> build.lock
flock -n 9 || exit 1
sha256sum --check recipe-inputs.sha256
if [[ -f preflight-build.log ]]; then cp preflight-build.log "preflight-build.$(date +%Y%m%d-%H%M%S).log"; fi
rm -f preflight-exit.txt
hostname > preflight.host
nohup bash "$root/preflight-build.sh" > preflight-build.log 2>&1 < /dev/null &
printf '%s\n' "$!" > preflight.pid
printf 'Started allocator/linkage preflight: host=%s pid=%s\n' "$(cat preflight.host)" "$(cat preflight.pid)"
