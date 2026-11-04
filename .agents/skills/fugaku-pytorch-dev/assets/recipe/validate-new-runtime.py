"""Run only after the new wheel is built; no imports from the baseline wheel."""
import json
import os
from pathlib import Path
import sys

assert not os.environ.get('LD_PRELOAD')
assert not os.environ.get('LD_LIBRARY_PATH')
assert not os.environ.get('PYTHONHOME')
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'wheel-install'),str(root/'runtime-site')]
import torch
import torch.distributed as dist
assert Path(torch.__file__).resolve().is_relative_to((root/'wheel-install').resolve())
assert torch.get_num_threads()==12
dist.init_process_group('mpi')
try:
    rank,world=dist.get_rank(),dist.get_world_size()
    data=torch.empty(64*1024*1024,dtype=torch.float32)
    data.fill_(1)
    ptr=data.data_ptr()
    mapping=[]
    active=False
    for line in Path('/proc/self/smaps').read_text().splitlines():
        first=line.split()[0]
        if '-' in first and ':' not in first:
            lo,hi=[int(x,16) for x in first.split('-')]
            active=lo<=ptr<hi
        if active:mapping.append(line)
    assert any(line.startswith('KernelPageSize:') and line.split()[1]=='2048' for line in mapping),mapping
    assert any(line.startswith('Private_Hugetlb:') and int(line.split()[1])>0 for line in mapping),mapping
    a=torch.ones((384,384))
    b=torch.full((384,384),2.)
    assert torch.equal(torch.mm(a,b),torch.full((384,384),768.))
    t=torch.tensor([float(rank+1)])
    work=dist.all_reduce(t,async_op=True)
    work.wait()
    assert t.item()==world*(world+1)/2
    maps=Path('/proc/self/maps').read_text()
    assert 'libmpg.so' in maps and 'libfjlapackexsve.so' in maps and 'libfjomp.so' in maps
    assert not any(name in maps for name in ['libomp.so','libgomp.so','libiomp'])
    print(json.dumps(dict(event='fresh_runtime_pass',rank=rank,world=world,torch=torch.__file__,
        threads=torch.get_num_threads(),tensor_mapping=mapping,maps=maps)),flush=True)
finally:
    dist.destroy_process_group()
