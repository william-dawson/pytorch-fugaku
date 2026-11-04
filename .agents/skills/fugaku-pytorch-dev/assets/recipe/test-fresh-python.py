"""Select the fresh wheel explicitly, then run a correctness suite."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
assert not any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONHOME', 'PYTHONPATH'))
sys.path[:0] = [str(root / 'wheel-install'), str(root / 'runtime-site')]
import torch
assert Path(torch.__file__).resolve().is_relative_to((root / 'wheel-install').resolve())
suite = sys.argv[1]
assert suite in ('python-smoke.py', 'python-async.py', 'python-benchmark.py')
runpy.run_path(str(root / suite), run_name='__main__')
