"""Check a newly prepared offline tree without compiling or contacting Fugaku."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tempfile

SKILL = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('workspace', type=Path)
parser.add_argument('--report', type=Path)
args = parser.parse_args()
root = args.workspace.resolve()
lock = json.loads((root / 'source-lock.json').read_text())
assert lock == json.loads((SKILL / 'assets/inputs.lock.json').read_text())
assert not (root / 'validation-results.json').exists(), 'Historical PASS evidence must not be staged'
assert not (root / 'target').exists(), 'Use a new, not previously built stage'
assert not list((root / 'src/pytorch').rglob('.git'))
sealed = 0
for line in (root / 'recipe-inputs.sha256').read_text().splitlines():
    sha, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    sealed += 1
for name, sha in json.loads((SKILL / 'assets/recipe.sha256.json').read_text()).items():
    assert hashlib.sha256((SKILL / 'assets/recipe' / name).read_bytes()).hexdigest() == sha, name
for path in list(root.glob('*.sh')):
    subprocess.run(['bash', '-n', str(path)], check=True)
for path in root.glob('*.py'):
    ast.parse(path.read_text())
patches = sorted(root.glob('*.patch'))
targets = set()
for patch in patches:
    for name in re.findall(r'^\+\+\+ b/(.+)$', patch.read_text(), re.MULTILINE):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        targets.add(name)
with tempfile.TemporaryDirectory(prefix='fugaku-patch-preflight-') as temporary:
    tree = Path(temporary)
    originals = {}
    for name in targets:
        source = root / 'src/pytorch' / name
        destination = tree / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        originals[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for patch in patches:
        subprocess.run(['git', '-C', str(tree), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(tree), 'apply', str(patch)], check=True)
    for patch in reversed(patches):
        subprocess.run(['git', '-C', str(tree), 'apply', '--reverse', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(tree), 'apply', '--reverse', str(patch)], check=True)
    assert all(hashlib.sha256((tree / name).read_bytes()).hexdigest() == sha for name, sha in originals.items())
extensions = list((root / 'runtime-site/numpy/_core').glob('_multiarray_umath*.so'))
assert len(extensions) == 1
header = extensions[0].read_bytes()[:20]
assert header[:4] == b'\x7fELF' and struct.unpack_from('<H', header, 18)[0] == 183
report = {'kind': 'local-offline-preflight', 'host': platform.platform(),
          'source_commit': lock['pytorch']['commit'], 'dependency_pins': len(lock['cpu_dependencies']),
          'sealed_inputs_verified': sealed, 'patches_applied_and_reversed': len(patches),
          'shell_syntax': 'PASS', 'python_syntax': 'PASS', 'target_numpy_elf': 'AArch64',
          'historical_pass_records_absent': True, 'target_compilation_performed': False,
          'fresh_cluster_reproduction_performed': False}
if args.report:
    args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
