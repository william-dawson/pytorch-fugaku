"""Assemble CMake-built AArch64 artifacts into a standard CPython 3.11 wheel."""
from pathlib import Path
import shutil
import re
from wheel.wheelfile import WheelFile

root = Path(__file__).resolve().parent
stage = root / 'python-package'
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir()
source = root / 'src/pytorch'
for name in ('torch', 'torchgen', 'functorch'):
    def ignore(directory, names):
        excluded = {'__pycache__', '.git'}
        if Path(directory) == source / name:
            excluded |= {'csrc', 'lib', 'test'}
        return set(names) & excluded
    shutil.copytree(source / name, stage / name, ignore=ignore)
torch = stage / 'torch'
(torch / 'lib').mkdir(exist_ok=True)
for prefix in ('target', 'python-prefix'):
    for path in (root / prefix / 'lib').glob('*.so*'):
        shutil.copy2(path, torch / 'lib' / path.name)
shutil.copytree(root / 'target/include', torch / 'include', dirs_exist_ok=True)
shutil.copytree(root / 'python-prefix/bin', torch / 'bin', dirs_exist_ok=True)
shutil.copy2(root / 'python-extension-output/_C.cpython-311-aarch64-linux-gnu.so', torch)
assert (torch / 'version.py').is_file()
version_file = torch / 'version.py'
version_file.write_text(re.sub(r"^git_version = .*", "git_version = '70d99e998b4955e0049d13a98d77ae1b14db1f45'", version_file.read_text(), flags=re.MULTILINE))
assert (torch / 'lib/libtorch_python.so').is_file()
assert (torch / 'bin/torch_shm_manager').is_file()
metadata = stage / 'torch-2.11.0+fugaku.llvm.dist-info'
metadata.mkdir()
requirements = ['filelock', 'typing-extensions>=4.15.0', 'setuptools<82', 'sympy>=1.13.3',
                'networkx>=2.5.1', 'jinja2', 'fsspec>=0.8.5', 'numpy>=2.0']
(metadata / 'METADATA').write_text(
    'Metadata-Version: 2.1\nName: torch\nVersion: 2.11.0+fugaku.llvm\n'
    'Summary: PyTorch for Fugaku A64FX with Fujitsu MPI and BLAS\nRequires-Python: >=3.11\n'
    + ''.join(f'Requires-Dist: {item}\n' for item in requirements) + '\n')
(metadata / 'WHEEL').write_text('Wheel-Version: 1.0\nGenerator: fugaku-cmake-cross\n'
                              'Root-Is-Purelib: false\nTag: cp311-cp311-linux_aarch64\n')
(metadata / 'licenses').mkdir()
shutil.copy2(source / 'LICENSE', metadata / 'licenses/LICENSE')
dist = root / 'dist'
dist.mkdir(exist_ok=True)
wheel = dist / 'torch-2.11.0+fugaku.llvm-cp311-cp311-linux_aarch64.whl'
with WheelFile(wheel, 'w') as archive:
    archive.write_files(str(stage))
print(f'Wheel: {wheel} ({wheel.stat().st_size} bytes)', flush=True)
