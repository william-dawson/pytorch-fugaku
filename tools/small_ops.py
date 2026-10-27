"""Small-tensor ops of OmniLearned's data path (150-particle jets), timed per
call: the 2c profile showed nonzero on 150 elements at 6.5 ms and copy_ on
[150, 4] at 4 ms, and its DataLoader iteration 9x the stock wheel's.

python small_ops.py [label]
"""
import sys
import time

import torch

label = sys.argv[1] if len(sys.argv) > 1 else "?"
print(f"== {label}: torch {torch.__version__}, capability {torch.backends.cpu.get_cpu_capability()}, "
      f"threads {torch.get_num_threads()}")
torch.manual_seed(0)
f = torch.randn(150, 4)
f8 = torch.randn(8, 150, 4)
m = f[:, 0] > 0
i64 = torch.randint(0, 150, (150,))
big = torch.randn(8, 150, 150)
ops = {
    "nonzero bool[150]": lambda: m.nonzero(),
    "all bool[150]": lambda: m.all(),
    "sum bool[150]": lambda: m.sum(),
    "copy_ f32[150,4]": lambda: torch.empty(150, 4).copy_(f),
    "to f64 [150,4]": lambda: f.to(torch.float64),
    "to i64 [150]": lambda: f[:, 0].to(torch.int64),
    "index f32[150,4][i64]": lambda: f[i64],
    "masked f32[150,4][bool]": lambda: f[m],
    "cat 8x[150,4]": lambda: torch.cat([f] * 8),
    "stack 8x[150,4]": lambda: torch.stack([f] * 8),
    "unsqueeze+view": lambda: f.unsqueeze(0).view(1, 600),
    "empty [150,4]": lambda: torch.empty(150, 4),
    "add f32[150,4]": lambda: f + f,
    "log f32[150,4]": lambda: f.abs().log(),
    "atan2 f32[150]": lambda: torch.atan2(f[:, 0], f[:, 1]),
    "norm f32[8,150,4]": lambda: f8.norm(dim=-1),
    "softmax f32[8,150,150]": lambda: big.softmax(-1),
    "where f32[150,4]": lambda: torch.where(f > 0, f, 0.0),
}
for name, op in ops.items():
    for _ in range(20):
        op()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 0.3:
        op()
        n += 1
    print(f"   {name:26s} {(time.perf_counter() - t0) / n * 1e6:10.1f} us/call")
