"""Per case: steady step time (max over ranks per step, median over steps after warmup),
losses, and fjmpi_async's blocked time per steady step (mean over ranks)."""
import json, statistics, sys
from pathlib import Path
for job in sys.argv[1:]:
    for case in sorted(d.name for d in Path(job).iterdir() if d.is_dir()):
        files = sorted(Path(job, case).glob('timing.rank*.json'))
        if not files:
            continue
        runs = [json.loads(f.read_text()) for f in files]
        n = min(len(r['steps']) for r in runs)
        per_step = [max(r['steps'][i]['wall_s'] for r in runs) for i in range(n)]
        warm = int(runs[0].get('warmup', 0)) or {12: 4, 16: 6}.get(n, 3)
        steady = per_step[warm:]
        losses = [round(s['loss'], 4) for s in runs[0]['steps']]
        line = f"{job} {case:10s} ranks={len(runs):2d} steady_med={statistics.median(steady):7.3f}s mean={statistics.mean(steady):7.3f}s steps={[round(x,2) for x in per_step]}"
        print(line)
        print(f"      losses={losses}")
        if 'comm' in runs[0]['steps'][0]:
            keys = ['wait_s', 'drain_s', 'gather_sync_s', 'reduce_scatter_s', 'allreduce_s', 'other_s']
            avg = {k: statistics.mean(statistics.mean(r['steps'][i]['comm'][k] for i in range(warm, n)) for r in runs) for k in keys}
            c = runs[0]['steps'][warm]['comm']
            print('      blocked/step (mean over ranks): ' + ' '.join(f"{k}={v:.3f}" for k, v in avg.items())
                  + f" total={sum(avg.values()):.3f} fill={c.get('chunk_used_bytes',0)/max(1,c.get('chunk_capacity_bytes',0)):.2f}" + f" | rank0 step{warm}: gathers_async={c['gathers_async']} sync={c['gathers_sync']} chunks={c['chunks']} waits={c['waits']} already_done={c['waits_already_done']} GB={c['gather_bytes']/1e9:.2f}")
