#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
trap 'code=$?; printf "%s\n" "$code" > preflight-exit.txt' EXIT
module load LLVM/llvmorg-22.1.0
compiler=/vol0004/apps/r/OSS_CN/llvm-22.1.0/cross_clangfx/clang-comp/bin/clang
mmm=/opt/FJSVxos/devkit/aarch64/rfs/opt/FJSVxos/mmm
fj=/opt/FJSVxtclanga/tcsds-1.2.43/lib64
python_prefix=/vol0402/data/ra000009/u10035/miniconda3
mkdir -p bin tmp
export TMPDIR="$root/tmp"
"$compiler" -v -O3 -mcpu=a64fx -msve-vector-bits=512 -no-pie \
  -I"$python_prefix/include/python3.11" python-runner.c \
  -L"$mmm/lib64" -Wl,--no-as-needed -lmpg -L"$fj" -lfjomp -lfjomphk -lfj90i -lfj90f -lfjsrcinfo -lfjcrt -lfjompcrt -lelf -Wl,--as-needed \
  -Wl,-T,"$mmm/util/bss-2mb.lds" -L"$python_prefix/lib" -lpython3.11 \
  -Wl,-rpath,/opt/FJSVxos/mmm/lib64 -Wl,-rpath,"$fj" -Wl,-rpath,"$python_prefix/lib" \
  -o bin/python3.11-fugaku
readelf -h bin/python3.11-fugaku > preflight-elf.txt
readelf -d bin/python3.11-fugaku > preflight-dynamic.txt
sha256sum bin/python3.11-fugaku > preflight-binary.sha256
