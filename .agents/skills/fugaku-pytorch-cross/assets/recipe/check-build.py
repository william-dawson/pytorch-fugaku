"""Reject silently disabled MPI/OpenMP and unintended OpenMP runtimes."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
cache = {}
for line in (root / "build-pytorch/CMakeCache.txt").read_text().splitlines():
    if line and not line.startswith(("#", "//")) and "=" in line and ":" in line.split("=", 1)[0]:
        key, value = line.split("=", 1)
        cache[key.split(":", 1)[0]] = value
for key in ("USE_DISTRIBUTED", "USE_MPI", "USE_C10D_MPI", "USE_OPENMP"):
    if cache.get(key) not in ("ON", "TRUE", "1"):
        raise SystemExit(f"Required backend was disabled: {key}={cache.get(key)}")
if "fjlapackexsve" not in cache.get("BLAS_LIBRARIES", ""):
    raise SystemExit("Fujitsu threaded BLAS was not selected")
if cache.get("FUGAKU_LAPACK_LINKS") not in ("ON", "TRUE", "1"):
    raise SystemExit("Fujitsu LAPACK link validation failed")
commands = json.loads((root / "build-pytorch/compile_commands.json").read_text())
functionalization = [x for x in commands if "/RegisterFunctionalization_" in x["file"]]
if not functionalization or any("-Wno-error=missing-prototypes" not in x["command"] for x in functionalization):
    raise SystemExit("Generated functionalization files lack the intended warning exemption")
mpi = [x for x in commands if x["file"].endswith("/ProcessGroupMPI.cpp")]
if not mpi or not any("USE_C10D_MPI" in x["command"] for x in mpi):
    raise SystemExit("PyTorch's ProcessGroupMPI is missing from the build")
if sys.argv[-1] == "artifacts":
    for name in ("libc10.so", "libtorch.so", "libtorch_cpu.so"):
        path = root / "target/lib" / name
        header = subprocess.check_output(["readelf", "-h", str(path)], text=True)
        if "AArch64" not in header:
            raise SystemExit(f"Wrong architecture: {path}")
        dynamic = subprocess.check_output(["readelf", "-d", str(path)], text=True)
        if name == "libtorch_cpu.so" and "[libfjlapackexsve.so" not in dynamic:
            raise SystemExit("libtorch_cpu does not retain its Fujitsu BLAS dependency")
        if any(f"[{lib}" in dynamic for lib in ("libomp.so", "libgomp.so", "libiomp")):
            raise SystemExit(f"Unexpected OpenMP runtime in {path}")
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(root / "target/lib/libtorch_cpu.so")], text=True)
    if not any("ProcessGroupMPI" in line and "createProcessGroupMPI" in line for line in symbols.splitlines()):
        raise SystemExit("Installed libtorch_cpu lacks the MPI backend factory")
print("Required MPI/OpenMP backends and Fujitsu BLAS verified", flush=True)
