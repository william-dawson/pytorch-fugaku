#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
compiler=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifrtpx
"$compiler" -show
"$compiler" -Kfast -Kopenmp -SSL2BLAMP vendor-reference.f90 -o vendor-reference
file vendor-reference
readelf -d vendor-reference
