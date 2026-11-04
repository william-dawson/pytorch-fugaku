#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
exec 9> build.lock
flock -n 9 || { echo 'Another build holds this workspace lock'; exit 1; }
sha256sum --check recipe-inputs.sha256
[[ -f src/pytorch/CMakeLists.txt && -f source-lock.json ]]
if [[ -f build.log ]]; then cp build.log "build.$(date +%Y%m%d-%H%M%S).log"; fi
rm -f exit-code.txt
hostname > build.host
nohup bash "$root/build.sh" > build.log 2>&1 < /dev/null &
printf '%s\n' "$!" > build.pid
printf 'Started LibTorch build: host=%s pid=%s\n' "$(cat build.host)" "$(cat build.pid)"
