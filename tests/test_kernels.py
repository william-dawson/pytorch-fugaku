"""torch_fugaku's kernels against the stock ones and an FP64 reference.

Runs with TORCH_FUGAKU=0 so the stock kernels can be measured first in the
same process. Fails on: accuracy beyond the measured bounds, inf/NaN
behaviour differing from stock, the non-contiguous/float64 fallback not
being bit-identical to stock, bmm (and its gradients) not matching a
slice-by-slice product.
"""
import sys

import torch

import torch_fugaku

failures = []


def check(cond, what):
    print(f"   {'ok  ' if cond else 'FAIL'} {what}", flush=True)
    if not cond:
        failures.append(what)


torch.manual_seed(0)
special = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e-8, -1e-8, 3.9, 3.94, 4.0, -4.0, 9.0, -9.02,
                        20.0, -20.0, 87.0, -87.0, 1e30, -1e30, float("inf"), float("-inf"), float("nan")])
x = torch.cat([torch.linspace(-12, 12, 400_001), torch.randn(100_000) * 3, special])
g = torch.randn_like(x)
d, gd = x.double(), g.double()
cdf = 0.5 * (1 + torch.erf(d * 0.7071067811865476))
pdf = torch.exp(-0.5 * d * d) * 0.3989422804014327
ref = {"gelu": d * cdf, "gelu_backward": gd * (cdf + d * pdf), "tanh": torch.tanh(d)}


def run():
    return {"gelu": torch.ops.aten.gelu(x), "gelu_backward": torch.ops.aten.gelu_backward(g, x),
            "tanh": torch.tanh(x)}


def ulp_err(out, r):
    r32 = r.float()
    ulp = (torch.nextafter(r32.abs(), torch.tensor(float("inf"))) - r32.abs()).double()
    finite = torch.isfinite(r32) & torch.isfinite(out)
    cond = finite & (r.abs() >= 1e-2)
    return ((out.double() - r).abs()[cond] / ulp[cond]).max().item()


nc = torch.randn(512, 513)[:, :512].t()
f64 = torch.randn(4096, dtype=torch.float64)
fallback = lambda: (torch.ops.aten.gelu(nc), torch.tanh(nc), torch.ops.aten.gelu(f64), torch.tanh(f64))
stock, fb_stock = run(), fallback()
bx = torch.randn(512, 83, 32, dtype=torch.float64, requires_grad=True)
by = torch.randn(512, 32, 83, dtype=torch.float64, requires_grad=True)
bref = torch.bmm(bx, by)
bgx, bgy = torch.autograd.grad(bref.sum(), (bx, by))

torch_fugaku.enable()
st = torch_fugaku.status()
print(f"torch {torch.__version__}, torch_fugaku {st}")
check("parbmm" in st["kernels"], "bmm override loaded")
if st["sve"]:
    check("fastmath" in st["kernels"], "SVE kernels loaded on an SVE CPU")
    fast = run()
    # Measured maxima (README): gelu 89, gelu_backward 121, tanh 2.5 ULP; allow a margin.
    bounds = {"gelu": 100, "gelu_backward": 140, "tanh": 3.0}
    for k, r in ref.items():
        e = ulp_err(fast[k], r)
        check(e <= bounds[k], f"{k}: {e:.2f} ULP (bound {bounds[k]}, stock {ulp_err(stock[k], r):.2f})")
        diff = (torch.isfinite(fast[k]) != torch.isfinite(stock[k])) | (torch.isnan(fast[k]) != torch.isnan(stock[k]))
        if k == "gelu_backward":
            # At x = +-inf the formula is inf * 0: stock gives NaN on A64FX and
            # +-inf on Neoverse, so the result there is implementation-defined.
            diff &= ~torch.isinf(x)
        check(not bool(diff.any()), f"{k}: inf/NaN as stock")
        for i in diff.nonzero().flatten()[:8].tolist():
            print(f"        x={x[i].item():.6g} g={g[i].item():.6g}: torch_fugaku {fast[k][i].item():.6g}, "
                  f"stock {stock[k][i].item():.6g}, FP64 {ref[k][i].item():.6g}")
    check(all(torch.equal(a, b) for a, b in zip(fb_stock, fallback())),
          "non-contiguous float32 and float64 take the stock kernel, bit for bit")
else:
    check("fastmath" not in st["kernels"], "no SVE kernels on a CPU without SVE")
out = torch.bmm(bx, by)
hx, hy = torch.autograd.grad(out.sum(), (bx, by))
check(torch.allclose(out, bref) and torch.allclose(hx, bgx) and torch.allclose(hy, bgy), "bmm and its gradients")
a, b = torch.randn(64, 83, 83), torch.randn(64, 83, 32)
check(torch.allclose(torch.bmm(a, b), torch.stack([a[i] @ b[i] for i in range(64)]), atol=1e-4),
      "bmm float32 against slice-by-slice")
check(not torch.backends.mkldnn.enabled, "oneDNN off")
print("PASS" if not failures else f"FAIL: {failures}")
sys.exit(1 if failures else 0)
