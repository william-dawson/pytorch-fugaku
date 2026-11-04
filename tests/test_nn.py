"""torch_fugaku.nn.ChunkedLMLoss against lm_head + F.cross_entropy: loss and both gradients, float64 and float32,
with ignored (-100) targets and a chunk that does not divide the token count."""
import sys

import torch
import torch.nn.functional as F

from torch_fugaku.nn import ChunkedLMLoss

torch.manual_seed(0)
ok = True
# Both paths take float32 logits (as transformers does), so float64 agrees to float32 rounding.
for dtype, tol in ((torch.float64, 1e-6), (torch.float32, 2e-5)):
    T, H, V = 1000, 64, 3000
    h0 = torch.randn(T, H, dtype=dtype)
    w0 = torch.randn(V, H, dtype=dtype) * 0.05
    y = torch.randint(0, V, (T,))
    y[::7] = -100
    h1, w1 = h0.clone().requires_grad_(), w0.clone().requires_grad_()
    ref = F.cross_entropy((h1 @ w1.t()).float(), y, ignore_index=-100)
    (ref * 3).backward()
    h2, w2 = h0.clone().requires_grad_(), w0.clone().requires_grad_()
    got = ChunkedLMLoss.apply(h2, w2, y, 96)
    (got * 3).backward()
    errs = [abs(got.item() - ref.item()) / abs(ref.item()),
            ((h2.grad - h1.grad).abs().max() / h1.grad.abs().max()).item(),
            ((w2.grad - w1.grad).abs().max() / w1.grad.abs().max()).item()]
    good = all(e < tol for e in errs)
    ok &= good
    print({"dtype": str(dtype), "loss": ref.item(), "rel_err_loss_gh_gw": errs, "ok": good})
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
