"""Regression checks for refusing incomplete or misleading job evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('record-validation.py')
PASS = 'GEMM=PASS LAPACK=PASS autograd=PASS ProcessGroupMPI=PASS OpenMP=PASS'

class ValidationGates(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copy2(SCRIPT, self.root / SCRIPT.name)
        (self.root / 'source-lock.json').write_text(json.dumps({'pytorch': {'commit': 'test-pin'}}))
        (self.root / 'exit-code.txt').write_text('0\n')
        (self.root / 'validation-exit.123.txt').write_text('0\n')
        self.output = self.root / 'output.123/0/1'
        self.output.mkdir(parents=True)
        (self.output / 'stdout.1.0').write_text(f'rank=0 host=node0 {PASS}\n')
        (self.output / 'stdout.1.1').write_text(f'rank=1 host=node1 {PASS}\n')

    def run_recorder(self):
        return subprocess.run([sys.executable, str(self.root / SCRIPT.name), 'libtorch', '123'], capture_output=True)

    def test_complete_two_host_job_passes(self):
        self.assertEqual(self.run_recorder().returncode, 0)
        record = json.loads((self.root / 'validation-results.json').read_text())
        self.assertEqual({r['host'] for r in record['ranks']}, {'node0', 'node1'})

    def test_exit_zero_with_partial_output_is_rejected(self):
        (self.output / 'stdout.1.1').write_text('rank=1 host=node1 GEMM=PASS\n')
        self.assertNotEqual(self.run_recorder().returncode, 0)
        self.assertFalse((self.root / 'validation-results.json').exists())

    def test_nonzero_exit_with_pass_markers_is_rejected(self):
        (self.root / 'validation-exit.123.txt').write_text('143\n')
        self.assertNotEqual(self.run_recorder().returncode, 0)

    def test_same_host_is_not_two_node_validation(self):
        (self.output / 'stdout.1.1').write_text(f'rank=1 host=node0 {PASS}\n')
        self.assertNotEqual(self.run_recorder().returncode, 0)

if __name__ == '__main__':
    unittest.main()
