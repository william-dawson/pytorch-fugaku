import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
cache = {}
for line in (root / "build-python/CMakeCache.txt").read_text().splitlines():
    if line and not line.startswith(("#", "//")) and "=" in line:
        key, value = line.split("=", 1)
        cache[key.split(":", 1)[0]] = value
for key in ("BUILD_PYTHON", "BUILD_LIBTORCHLESS", "USE_DISTRIBUTED", "USE_MPI", "USE_C10D_MPI", "USE_OPENMP", "USE_NUMPY"):
    assert cache.get(key) in ("ON", "TRUE", "1"), (key, cache.get(key))
expected = Path('/vol0402/data/ra000009/u10035/miniconda3/include/python3.11').resolve()
assert Path(cache['Python_INCLUDE_DIR']).resolve() == expected
commands = json.loads((root / 'build-python/compile_commands.json').read_text())
assert not any('CMakeFiles/torch_cpu.dir' in entry['command'] for entry in commands)
bindings = [entry for entry in commands if entry['file'].endswith('/distributed/c10d/init.cpp')]
assert bindings and all('USE_C10D_MPI' in entry['command'] for entry in bindings)
assert all('/fminiconda3/envs/front/include' not in entry['command'] for entry in commands)
if sys.argv[-1] == 'artifacts':
    for path in (root / 'python-package/torch/lib').glob('*.so*'):
        header = subprocess.check_output(['readelf', '-h', str(path)], text=True)
        assert 'AArch64' in header, path
        dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True)
        assert all('[' + lib not in dynamic for lib in ('libomp.so', 'libgomp.so', 'libiomp')), path
    extension = root / 'python-package/torch/_C.cpython-311-aarch64-linux-gnu.so'
    assert 'AArch64' in subprocess.check_output(['readelf', '-h', str(extension)], text=True)
print('Python 3.11 target headers, MPI bindings and imported LibTorch verified', flush=True)
