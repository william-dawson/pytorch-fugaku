#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
exec 9> build.lock
flock -n 9 || { echo 'Another build holds this workspace lock'; exit 1; }
sha256sum --check recipe-inputs.sha256
# A fresh two-node LibTorch validation must create this file; preparation never copies historical PASS records.
host-venv/bin/python -c 'import json; r=json.load(open("validation-results.json")); assert r["build_exit_code"] == 0 and r["application_exit_code"] == 0; assert len({x["host"] for x in r["ranks"]}) >= 2'
if [[ -f python-build.log ]]; then cp python-build.log "python-build.$(date +%Y%m%d-%H%M%S).log"; fi
rm -f python-exit-code.txt
hostname > python-build.host
nohup bash "$root/build-python.sh" > python-build.log 2>&1 < /dev/null &
printf '%s\n' "$!" > python-build.pid
printf 'Started Python build: host=%s pid=%s\n' "$(cat python-build.host)" "$(cat python-build.pid)"
