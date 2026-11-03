"""h2h-mlock-test/run_port.py for any node count, with the llvm variant's backend chosen by
H2H_BACKEND: fjmpi_port (../h2h-profile/port, the container's staged MPI backend rebuilt against
the llvm wheel) or fjmpi_async (../h2h-async/backend: fjmpi_port plus overlapped allgathers;
TORCH_FUGAKU_ASYNC=0 makes it synchronous). The container always uses its own fjmpi.
fjmpi_async's per-step blocked-time stats go into each step record as "comm".
H2H_PROFILE_STEPS=a,b: torch.profiler over steps a..b; operator tables in profile.rank{0,last}.txt.
H2H_TRACE_STEPS=a,b: fjmpi_async's event trace (TORCH_FUGAKU_TRACE=1) for steps a..b in
trace.rank<N>.json, with phase and step start times on the same clock (time.perf_counter)."""
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import runpy
import socket
import sys
import time

started = time.perf_counter()
rank = int(os.environ['PMIX_RANK'])
local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank % 4))
initial_affinity = sorted(os.sched_getaffinity(0))
assert len(initial_affinity) in (12, 48), initial_affinity
if len(initial_affinity) == 48:
    os.sched_setaffinity(0, initial_affinity[local_rank*12:(local_rank+1)*12])
assert len(os.sched_getaffinity(0)) == 12

if os.environ['H2H_VARIANT'] == 'llvm':
    fresh = Path('/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross-largepage/rebuild-largepage')
    # LD_PRELOAD only as the case's allocator (rank.sh H2H_LD_PRELOAD), nothing else.
    assert os.environ.get('LD_PRELOAD', '') == os.environ.get('H2H_LD_PRELOAD', '')
    assert not any(os.environ.get(k) for k in ('LD_LIBRARY_PATH', 'PYTHONHOME', 'PYTHONPATH'))
    sys.path[:0] = [str(fresh/'wheel-install'), str(fresh/'runtime-site'),
                   str(Path(__file__).resolve().parent.parent/'h2h-profile/port'),
                   os.environ.get('H2H_ASYNC_PATH', str(Path(__file__).resolve().parent.parent/'h2h-async/backend')),
                   str(Path(__file__).resolve().parent.parent/'h2h-tiny/site')]

import torch
import torch.distributed as dist
if os.environ['H2H_VARIANT'] == 'llvm':
    assert Path(torch.__file__).resolve().is_relative_to((fresh/'wheel-install').resolve())

variant = os.environ['H2H_VARIANT']
mode = os.environ['H2H_MODE']
backend = os.environ.get('H2H_BACKEND', 'fjmpi_port') if variant == 'llvm' else 'fjmpi'
assert backend in ('fjmpi', 'fjmpi_port', 'fjmpi_async'), backend
assert torch.__version__ == ('2.11.0+fugaku.llvm' if variant == 'llvm' else '2.11.0+cu130')
assert torch.get_num_threads() == 12
if variant == 'container':
    import torch_fugaku
    torch_fugaku.register_backend()
else:
    assert 'torch_fugaku' not in sys.modules
comm_stats = None
if backend == 'fjmpi_port':
    import fjmpi_port
    fjmpi_port.register_backend()
elif backend == 'fjmpi_async':
    import fjmpi_async
    fjmpi_async.register_backend()
    comm_stats = fjmpi_async
world = int(os.environ['WORLD_SIZE'])
dist.init_process_group(backend)
assert dist.get_rank() == rank and dist.get_world_size() == world
os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local_rank), LOCAL_SIZE='4')
warmup = int(os.environ['H2H_WARMUP'])
def step_range(name):
    v = os.environ.get(name)
    return tuple(int(x) for x in v.split(',')) if v else None
profile_steps, trace_steps = step_range('H2H_PROFILE_STEPS'), step_range('H2H_TRACE_STEPS')
profiler, traced = None, []

import deepspeed
from deepspeed.ops.op_builder.all_ops import ALL_OPS
from deepspeed.ops.op_builder.builder import OpBuilder
deepspeed.ops.__compatible_ops__ = dict.fromkeys(ALL_OPS, False)
def no_jit(self, *args, **kwargs):
    raise RuntimeError(f'Unplanned extension: {self.name}')
OpBuilder.load = OpBuilder.jit_load = no_jit

output = Path(os.environ['H2H_OUTPUT'])
output.mkdir(parents=True, exist_ok=True)
phases, steps, input_hashes = [], [], []
step = -1
window_start = None
window_end = None

def rss_gb(field):
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(field + ':'):
            return int(line.split()[1]) / 2**20
    return None

def reset_peak():
    # Linux: writing 5 resets VmHWM to the current RSS, so each phase gets its own peak. Opt-in
    # (H2H_PEAK=1): it walks the page tables every phase and skewed step times (job 53473983).
    if os.environ.get('H2H_PEAK') != '1':
        return
    try:
        Path('/proc/self/clear_refs').write_text('5')
    except OSError:
        pass

def snapshot():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return dict(process_cpu_s=time.process_time(), main_thread_cpu_s=time.thread_time(),
        user_cpu_s=u.ru_utime, system_cpu_s=u.ru_stime, minor_faults=u.ru_minflt,
        major_faults=u.ru_majflt, voluntary_switches=u.ru_nvcsw, involuntary_switches=u.ru_nivcsw)

def audit(label):
    maps = Path('/proc/self/maps').read_text()
    (output/f'maps.{label}.rank{rank}.txt').write_text(maps)
    (output/f'smaps.{label}.rank{rank}.txt').write_text(Path('/proc/self/smaps').read_text())
    tasks = []
    for p in Path('/proc/self/task').iterdir():
        try:
            tasks.append(dict(tid=p.name, status=p.joinpath('status').read_text(), schedstat=p.joinpath('schedstat').read_text()))
        except OSError:
            pass
    result = dict(label=label, variant=variant, mode=mode, rank=rank, host=socket.gethostname(),
        local_rank=local_rank, initial_affinity=initial_affinity, affinity=sorted(os.sched_getaffinity(0)),
        python=sys.version, executable=sys.executable, torch=torch.__version__, torch_path=torch.__file__,
        torch_config=torch.__config__.show(), parallel_info=torch.__config__.parallel_info(),
        packages={n:importlib.metadata.version(n) for n in ['deepspeed','transformers','numpy','accelerate']},
        backend=dist.get_backend(), mkldnn=torch.backends.mkldnn.enabled,
        autograd_multithreading=torch.autograd.is_multithreading_enabled(),
        environment={k:v for k,v in os.environ.items() if k.startswith(('OMP_','MALLOC_','MIMALLOC_','XOS_','TORCH_','PF_')) or k in ['LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH']},
        tasks=tasks, meminfo=Path('/proc/meminfo').read_text(), status=Path('/proc/self/status').read_text(),
        torch_fugaku=torch_fugaku.status() if variant == 'container' else None)
    # Hash installed artifacts after training only: no large-file scan to warm the baseline.
    if label == 'after' and rank in (0,4):
        paths = {line.split()[-1] for line in maps.splitlines() if '/' in line and '.so' in line}
        result['binary_hashes'] = {}
        for path in sorted(paths):
            if not any(x in path for x in ['libc10.', 'libtorch_cpu.', 'libmpi.', 'libfj', 'libmpg', 'libgomp', 'libomp.', 'torch_fugaku']):
                continue
            try:
                h = hashlib.sha256()
                with open(path, 'rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
                result['binary_hashes'][path] = h.hexdigest()
                if 'libc10.so' in path:
                    # Preserve the small allocator-containing artifact for local ELF inspection.
                    import shutil
                    shutil.copyfile(path, output/f'libc10.rank{rank}.so')
            except OSError as e:
                result['binary_hashes'][path] = str(e)
    (output/f'audit.{label}.rank{rank}.json').write_text(json.dumps(result, indent=2))

# Phase instrumentation is absent in the plain control. No Work wrapping or profiler.
if mode == 'resources':
    from deepspeed.runtime.engine import DeepSpeedEngine
    def wrap(fn, label):
        @functools.wraps(fn)
        def measured(*args, **kwargs):
            global step
            if label == 'forward': step += 1
            rss_start = rss_gb('VmRSS')
            reset_peak()
            before = snapshot()
            start = time.perf_counter()
            value = fn(*args, **kwargs)
            wall = time.perf_counter()-start
            after = snapshot()
            phases.append(dict(step=step, phase=label, start_s=start, wall_s=wall,
                rss_start_gb=rss_start, rss_end_gb=rss_gb('VmRSS'), rss_peak_gb=rss_gb('VmHWM'),
                **{k:after[k]-before[k] for k in before}))
            # Also to the log as it happens: a rank killed for memory never writes timing.json.
            print(json.dumps(dict(event='phase', **{k: phases[-1][k] for k in
                ('step', 'phase', 'wall_s', 'rss_start_gb', 'rss_end_gb', 'rss_peak_gb', 'minor_faults')})), flush=True)
            return value
        return measured
    for name in ['forward','backward','step']:
        setattr(DeepSpeedEngine, name, wrap(getattr(DeepSpeedEngine,name), name))

step_started = {}
def step_start(index):
    global window_start, profiler
    if index == warmup: window_start = time.perf_counter()
    if comm_stats:
        comm_stats.reset_stats()
        if trace_steps and hasattr(comm_stats, 'trace'):
            events = comm_stats.trace()  # clears; keep only the traced steps
            if trace_steps[0] < index <= trace_steps[1] + 1 and events:
                traced.extend(events)
    if profile_steps and index == profile_steps[0]:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        profiler.__enter__()
    step_started[index] = time.perf_counter()

def record_step(index, elapsed, loss):
    global window_end, profiler
    window_end = time.perf_counter()
    if trace_steps and comm_stats and index == trace_steps[1] and hasattr(comm_stats, 'trace'):
        traced.extend(comm_stats.trace())
        (output/f'trace.rank{rank}.json').write_text(json.dumps(dict(
            kinds=['issue','start','done','test_miss','wait_begin','wait_end','drain_begin','drain_end','coll_begin','coll_end'],
            events=traced, steps=step_started, phases=phases)))
    if profiler is not None and index == profile_steps[1]:
        profiler.__exit__(None, None, None)
        if rank in (0, world - 1):
            ka = profiler.key_averages()
            (output/f'profile.rank{rank}.txt').write_text(
                ka.table(sort_by='self_cpu_time_total', row_limit=80, max_name_column_width=70) + '\n\n' +
                ka.table(sort_by='cpu_time_total', row_limit=40, max_name_column_width=70))
        profiler = None
    steps.append(dict(step=index, wall_s=elapsed, loss=loss, **({'comm': comm_stats.stats()} if comm_stats else {})))

def record_input(ids):
    input_hashes.append(hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest())

def record_initial(engine):
    # ZeRO3 local shards, ordered by parameter name; before training and outside timers.
    h = hashlib.sha256()
    tensors = []
    for name, p in engine.module.named_parameters():
        h.update(name.encode())
        t = getattr(p, 'ds_tensor', p).detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
        tensors.append(dict(name=name,address=t.data_ptr(),bytes=t.numel()*t.element_size()))
    (output/f'initial.rank{rank}.json').write_text(json.dumps(dict(sha256=h.hexdigest(), scope='ordered local parameter shards',tensors=tensors)))
    (output/f'smaps.initial.rank{rank}.txt').write_text(Path('/proc/self/smaps').read_text())

try:
    audit('before')
    runpy.run_path(str(Path(__file__).with_name('llama_train.py')), run_name='__main__',
        init_globals={'h2h_record_step':record_step, 'h2h_record_input':record_input, 'h2h_record_initial':record_initial, 'h2h_step_start':step_start})
    (output/f'timing.rank{rank}.json').write_text(json.dumps(dict(variant=variant,backend=backend,world=world,mode=mode,rank=rank,
        steps=steps,phases=phases,input_hashes=input_hashes,window_wall_s=window_end-window_start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        elapsed_before_final_audit_s=time.perf_counter()-started),indent=2))
    audit('after')
    dist.barrier()
    print(json.dumps(dict(event='baseline_complete', rank=rank)), flush=True)
finally:
    dist.destroy_process_group()
