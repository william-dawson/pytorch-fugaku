"""Record only complete successful outputs from a new reproduction job."""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=['libtorch', 'smoke', 'async', 'benchmark'])
parser.add_argument('job_id')
parser.add_argument('--ranks', type=int, default=2)
args = parser.parse_args()
root = Path(__file__).resolve().parent
assert (root / f'validation-exit.{args.job_id}.txt').read_text().strip() == '0'
build_exit = 'exit-code.txt' if args.mode == 'libtorch' else 'python-exit-code.txt'
assert (root / build_exit).read_text().strip() == '0'
rows = []
for rank in ([0] if args.mode == 'benchmark' else range(args.ranks)):
    path = root / f'output.{args.job_id}/0/1/stdout.1.{rank}'
    text = path.read_text()
    if args.mode == 'libtorch':
        for token in ('GEMM=PASS', 'LAPACK=PASS', 'autograd=PASS', 'ProcessGroupMPI=PASS', 'OpenMP=PASS'):
            assert token in text, (path, token)
        host = re.search(r'\bhost=(\S+)', text).group(1)
        rows.append({'rank': rank, 'host': host, 'status': 'PASS'})
    elif args.mode == 'smoke':
        row = json.loads(next(line for line in text.splitlines() if line.startswith('{')))
        assert row['rank'] == rank
        assert all(row[key] == 'PASS' for key in ('GEMM', 'LAPACK', 'autograd', 'NumPy', 'MPI', 'OpenMP'))
        rows.append(row)
    elif args.mode == 'async':
        row = json.loads((root / f'async-results.{args.job_id}.rank{rank}.json').read_text())
        assert row['rank'] == rank and row['status'] == 'PASS'
        assert len(row['cases']) == 7 and all(case['status'] == 'PASS' for case in row['cases'])
        rows.append(row)
    elif rank == 0:
        rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
        assert len(rows) == 2 and all(row['correctness'] == 'PASS' for row in rows)
        assert all(row['ranks'] == args.ranks and row['n'] == 4096 for row in rows)
if args.mode != 'benchmark':
    assert len({row['host'] for row in rows}) >= 2
record = {'job_id': args.job_id, 'mode': args.mode, 'build_exit_code': 0,
          'application_exit_code': 0, 'ranks': rows,
          'source_commit': json.loads((root / 'source-lock.json').read_text())['pytorch']['commit']}
name = 'validation-results.json' if args.mode == 'libtorch' else f'validation-{args.mode}.{args.job_id}.json'
(root / name).write_text(json.dumps(record, indent=2) + '\n')
print(f'Validated {args.mode}: {name}')
