"""Fetch pinned public sources and wheels; never runs on a compute node."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SKILL = Path(__file__).resolve().parents[1]
LOCK = json.loads((SKILL / 'assets/inputs.lock.json').read_text())

def run(*args):
    subprocess.run(args, check=True)

def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()

def verify(path, expected):
    assert revision(path) == expected, f'Wrong revision: {path}'

def wheels(path, entries, architecture, version):
    path.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, sha in entries.items():
        file = path / name
        if file.exists():
            assert hashlib.sha256(file.read_bytes()).hexdigest() == sha, f'Checksum mismatch: {file}'
        else:
            package, release = name.split('-')[:2]
            missing.append(f'{package}=={release}')
    if missing:
        run(sys.executable, '-m', 'pip', 'download', '--dest', str(path), '--only-binary=:all:',
            '--no-deps', '--python-version', version, '--implementation', 'cp', '--abi', 'cp' + version,
            '--platform', 'manylinux2014_' + architecture, '--platform', 'manylinux_2_17_' + architecture, *missing)
    for name, sha in entries.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == sha, name

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    args = parser.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / 'source'
    if not source.exists():
        run('git', 'clone', '--depth', '1', '--branch', LOCK['pytorch']['ref'], LOCK['pytorch']['url'], str(source))
    verify(source, LOCK['pytorch']['commit'])
    modules = {k: v for k, v in LOCK['cpu_dependencies'].items() if k != 'eigen'}
    run('git', '-C', str(source), 'submodule', 'update', '--init', '--depth', '1', '--',
        *['third_party/' + name for name in modules])
    for name, sha in modules.items():
        verify(source / 'third_party' / name, sha)
    eigen = source / 'third_party/eigen'
    if not eigen.exists() or not any(eigen.iterdir()):
        run('git', 'clone', '--depth', '1', '--branch', LOCK['eigen']['ref'], LOCK['eigen']['url'], str(eigen))
    verify(eigen, LOCK['eigen']['commit'])
    wheels(cache / 'wheels', LOCK['host_wheels'], 'x86_64', '312')
    wheels(cache / 'host311-wheels', LOCK['host311_wheels'], 'x86_64', '311')
    wheels(cache / 'python-wheels', LOCK['target_wheels'], 'aarch64', '311')
    print(f'Pinned input cache ready: {cache}')

if __name__ == '__main__':
    main()
