"""Create a fresh, offline-capable build tree from verified inputs, not prior artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

SKILL = Path(__file__).resolve().parents[1]
LOCK = json.loads((SKILL / 'assets/inputs.lock.json').read_text())

def verify_revision(path, expected):
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError(f'Wrong source revision at {path}: {actual}, expected {expected}')

def export(path, revision, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # git archive exports committed content, excluding unrelated worktree changes.
    with subprocess.Popen(['git', '-C', str(path), 'archive', '--format=tar', revision], stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            archive.extractall(destination, filter='data')
        if process.wait() != 0:
            raise RuntimeError(f'git archive failed: {path}')

def copy_wheels(source, target, entries):
    target.mkdir(parents=True, exist_ok=True)
    for name, sha in entries.items():
        file = source / name
        if hashlib.sha256(file.read_bytes()).hexdigest() != sha:
            raise ValueError(f'Checksum mismatch: {file}')
        shutil.copy2(file, target / name)

def job_spec(root, project, mode, ranks, nodes):
    ppn = ranks // nodes
    threads = 12 if mode == 'async' else 48 // ppn
    return {'name': 'cross-' + mode + '-r' + str(ranks), 'directory': root,
            'executable': '/bin/bash', 'arguments': ['./run-validation.sh', mode, str(ranks)], 'launcher': 'env',
            'attributes': {'account': project, 'queue_name': 'f-pt', 'duration': '00:05:00',
                           'custom_attributes': {'gfscache_volume': '/vol0004', '--mpi': f'max-proc-per-node={ppn}',
                                                 '-L': 'jobenv=linux2,freq=2000,eco_state=0'}},
            'resources': {'node_count': nodes, 'processes_per_node': ppn, 'cpu_cores_per_process': threads}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--cache', type=Path, help='Cache made by fetch-inputs.py')
    for name in ('source', 'host-wheels', 'host311-wheels', 'target-wheels'):
        parser.add_argument('--' + name, type=Path)
    for name in ('remote-root', 'project', 'host-python312', 'host-python311', 'target-python-prefix'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--fujitsu-root', default='/opt/FJSVxtclanga/tcsds-1.2.43')
    parser.add_argument('--llvm-root', default='/vol0004/apps/r/OSS_CN/llvm-22.1.0')
    parser.add_argument('--sysroot', default='/opt/FJSVxos/devkit/aarch64/rfs')
    args = parser.parse_args()
    for name in ('remote_root', 'project', 'host_python312', 'host_python311', 'target_python_prefix', 'fujitsu_root', 'llvm_root', 'sysroot'):
        if not re.fullmatch(r'[A-Za-z0-9_./+\-]+', getattr(args, name)):
            parser.error(f'{name} must be a plain path or identifier without whitespace or shell syntax')
    cache = args.cache or Path('.')
    source = args.source or cache / 'source'
    host_wheels = args.host_wheels or cache / 'wheels'
    host311_wheels = args.host311_wheels or cache / 'host311-wheels'
    target_wheels = args.target_wheels or cache / 'python-wheels'
    # Verify all inputs before creating the new workspace.
    verify_revision(source, LOCK['pytorch']['commit'])
    for name, revision in LOCK['cpu_dependencies'].items():
        verify_revision(source / 'third_party' / name, revision)
    for directory, entries in [(host_wheels, LOCK['host_wheels']), (host311_wheels, LOCK['host311_wheels']), (target_wheels, LOCK['target_wheels'])]:
        for name, sha in entries.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha:
                raise ValueError(f'Checksum mismatch: {directory / name}')
    root = args.workspace.resolve()
    if root.exists():
        parser.error('Workspace must not exist; preparation never overlays a previous build')
    shutil.copytree(SKILL / 'assets/recipe', root)
    export(source, LOCK['pytorch']['commit'], root / 'src/pytorch')
    for name, revision in LOCK['cpu_dependencies'].items():
        export(source / 'third_party' / name, revision, root / 'src/pytorch/third_party' / name)
    copy_wheels(host_wheels, root / 'wheels', LOCK['host_wheels'])
    copy_wheels(host311_wheels, root / 'host311-wheels', LOCK['host311_wheels'])
    copy_wheels(target_wheels, root / 'python-wheels', LOCK['target_wheels'])
    replacements = {
        '/vol0402/data/ra000009/u10035/fminiconda3/bin/python3.12': args.host_python312,
        '/vol0402/data/ra000009/u10035/fminiconda3/envs/front/bin/python3.11': args.host_python311,
        '/vol0402/data/ra000009/u10035/miniconda3': args.target_python_prefix,
        '/opt/FJSVxtclanga/tcsds-1.2.43': args.fujitsu_root,
        '/vol0004/apps/r/OSS_CN/llvm-22.1.0': args.llvm_root,
        '/opt/FJSVxos/devkit/aarch64/bin/aarch64-linux-gnu-ld': str(Path(args.sysroot).parent / 'bin/aarch64-linux-gnu-ld'),
        '/opt/FJSVxos/devkit/aarch64/rfs': args.sysroot,
    }
    for path in root.rglob('*'):
        if path.is_file() and 'src' not in path.relative_to(root).parts and path.suffix in ('.sh', '.py', '.cmake', '.txt'):
            text = path.read_text()
            for old, new in replacements.items():
                text = text.replace(old, new)
            path.write_text(text)
    # Isolate generator dependencies rather than inheriting the original front env.
    path = root / 'build-python.sh'
    text = path.read_text().replace('import yaml; print(sys.version)', 'print(sys.version)')
    text = text.replace('venv --system-site-packages host-venv311', 'venv host-venv311')
    text = text.replace('--find-links wheels --no-deps typing-extensions', '--find-links wheels --find-links host311-wheels --no-deps PyYAML==6.0.2 numpy==2.2.6 typing-extensions')
    path.write_text(text)
    # Pin the release metadata even when the source export has no .git directory.
    path = root / 'build.sh'
    text = path.read_text().replace('export PYTHONNOUSERSITE=1', 'export PYTORCH_BUILD_NUMBER=1\nexport PYTHONNOUSERSITE=1')
    path.write_text(text)
    for name in ('launch.sh', 'launch-python.sh', 'record-validation.py', 'run-validation.sh'):
        shutil.copy2(SKILL / 'scripts' / name, root / name)
    (root / 'source-lock.json').write_text(json.dumps(LOCK, indent=2) + '\n')
    for mode, ranks, nodes in [('libtorch', 2, 2), ('smoke', 2, 2), ('async', 2, 2), ('benchmark', 1, 1), ('benchmark', 2, 1), ('benchmark', 4, 1)]:
        (root / f'job-{mode}-r{ranks}.json').write_text(json.dumps(job_spec(args.remote_root, args.project, mode, ranks, nodes), indent=2) + '\n')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
                    '--find-links', str(root / 'python-wheels'), '--find-links', str(root / 'wheels'),
                    '--target', str(root / 'runtime-site'), '--platform', 'manylinux2014_aarch64',
                    '--platform', 'manylinux_2_17_aarch64', '--python-version', '3.11', '--implementation', 'cp', '--abi', 'cp311',
                    '--only-binary=:all:', *[str(p) for p in sorted((root / 'python-wheels').glob('*.whl'))],
                    *[str(root / 'wheels' / name) for name in LOCK['packaging_wheels']]], check=True)
    # This manifest deliberately excludes source files changed by idempotent patches,
    # generated files and the checksum manifest itself. Evidence is not staged.
    inputs = [p for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] not in ('src', 'runtime-site')]
    (root / 'recipe-inputs.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n' for p in sorted(inputs)))
    assert not (root / 'validation-results.json').exists()
    print(f'Fresh offline build tree ready: {root}; transfer its contents to {args.remote_root}')

if __name__ == '__main__':
    main()
