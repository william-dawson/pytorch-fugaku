"""What a torch build gives OmniLearned's hot CPU ops on A64FX: build facts, the
accuracy of gelu/tanh and their backward against FP64 (in ULP of the float32
result, as ../../omnilearned-cpu-tuning/scripts/test_fastmath.py), their
speed, and bmm on many small slices. Run the same script on the stock wheel,
with and without the run-time overrides, and on each built wheel.

python check_build.py [label]
"""
import os
import sys
import time

import torch

label = sys.argv[1] if len(sys.argv) > 1 else "?"
print(f"== {label}: torch {torch.__version__}, capability {torch.backends.cpu.get_cpu_capability()}, "
      f"threads {torch.get_num_threads()}, mkldnn {torch.backends.mkldnn.is_available()}, "
      f"mpi {torch.distributed.is_mpi_available()}, gloo {torch.distributed.is_gloo_available()}")
blas = [l.strip() for l in torch.__config__.show().splitlines() if "BLAS_INFO" in l or "LAPACK_INFO" in l]
print("   " + "; ".join(blas))

torch.manual_seed(0)
special = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e-8, -1e-8, 3.9, 3.94, 4.0, -4.0, 9.0, -9.02,
                        20.0, -20.0, 87.0, -87.0, 1e30, -1e30, float("inf"), float("-inf"), float("nan")])
x = torch.cat([torch.linspace(-12, 12, 4_000_001), torch.randn(1_000_000) * 3, special])
g = torch.randn_like(x)
d, gd = x.double(), g.double()
cdf = 0.5 * (1 + torch.erf(d * 0.7071067811865476))
pdf = torch.exp(-0.5 * d * d) * 0.3989422804014327
t32 = torch.tanh(x)
ref = {"gelu": d * cdf, "gelu_backward": gd * (cdf + d * pdf), "tanh": torch.tanh(d),
       "tanh_backward": gd * (1 - t32.double() ** 2)}
out = {"gelu": torch.ops.aten.gelu(x), "gelu_backward": torch.ops.aten.gelu_backward(g, x),
       "tanh": torch.tanh(x), "tanh_backward": torch.ops.aten.tanh_backward(g, t32)}


def ulp_err(o, r):
    r32 = r.float()
    ulp = (torch.nextafter(r32.abs(), torch.tensor(float("inf"))) - r32.abs()).double()
    finite = torch.isfinite(r32) & torch.isfinite(o)
    cond = finite & (r.abs() >= 1e-2)
    ok = bool((torch.isfinite(r32) == torch.isfinite(o)).all() and (torch.isnan(r32) == torch.isnan(o)).all())
    return ((o.double() - r).abs()[cond] / ulp[cond]).max().item(), ok


def timed(f, reps=5):
    for _ in range(2):
        f()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t0) / reps * 1e3


n = 1 << 24
a, b = torch.randn(n), torch.randn(n)
tb = torch.tanh(a)
ops = {"gelu": lambda: torch.ops.aten.gelu(a), "gelu_backward": lambda: torch.ops.aten.gelu_backward(b, a),
       "tanh": lambda: torch.tanh(a), "tanh_backward": lambda: torch.ops.aten.tanh_backward(b, tb),
       "add": lambda: a + b, "mul_add": lambda: a * b + a}
for k, f in ops.items():
    if k in ref:
        u, ok = ulp_err(out[k], ref[k])
        acc = f"{u:8.2f} ULP  nonfinite {'ok' if ok else 'WRONG'}"
    else:
        acc = " " * 26
    ms = timed(f)
    gbs = (3 if k in ("gelu_backward", "tanh_backward", "add") else 6 if k == "mul_add" else 2) * 4 * n / ms / 1e6
    print(f"   {k:14s} {acc}  {ms:8.2f} ms  {gbs:6.1f} GB/s")

# bmm: many small slices, as OmniLearned's biased attention, and fewer larger ones.
for nb, m, k_, nn in [(4096, 32, 32, 32), (1024, 128, 32, 128), (64, 256, 256, 256)]:
    p, q = torch.randn(nb, m, k_), torch.randn(nb, k_, nn)
    exact = torch.equal(torch.bmm(p, q), torch.stack([p[i] @ q[i] for i in range(nb)]))
    ms = timed(lambda: torch.bmm(p, q))
    print(f"   bmm {nb}x{m}x{k_}x{nn:<4d}  {ms:8.2f} ms  {2 * nb * m * k_ * nn / ms / 1e6:7.1f} GFLOP/s  "
          f"slice-by-slice identical {exact}")

w = torch.randn(3840, 512)
y = torch.randn(512, 1536)
print(f"   sgemm 3840x1536x512   {timed(lambda: w @ y):8.2f} ms")
