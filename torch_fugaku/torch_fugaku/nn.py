"""A language-model loss that never holds the [tokens, vocab] logits.

transformers' causal-LM loss keeps the float32 logits and their gradient: at
seq 2048 and a 128k vocabulary that is ~3 GB per sequence, which decides
whether 4 ranks fit on a 32 GB Fugaku node. ``ChunkedLMLoss`` computes
lm_head and cross-entropy in token chunks with the gradients in the forward
(as Liger's fused linear cross-entropy), at the same FLOPs.
"""
import types

import torch

__all__ = ["ChunkedLMLoss", "chunked_lm_loss"]


class ChunkedLMLoss(torch.autograd.Function):
    """Mean cross-entropy of h @ w.T against y (-100: ignored), in chunks of ``chunk`` tokens."""

    @staticmethod
    def forward(ctx, h, w, y, chunk):
        valid = y != -100
        n = valid.sum().clamp(min=1).to(torch.float32)
        gh, gw = torch.empty_like(h), torch.zeros(w.shape, dtype=torch.float32)
        total = torch.zeros((), dtype=torch.float32)
        for s in range(0, h.shape[0], chunk):
            hc, yc, vc = h[s:s + chunk], y[s:s + chunk].clamp(min=0), valid[s:s + chunk]
            logits = (hc @ w.t()).float()
            lse = torch.logsumexp(logits, 1)
            total += ((lse - logits.gather(1, yc[:, None])[:, 0]) * vc).sum()
            p = logits.sub_(lse[:, None]).exp_()  # softmax, in place
            p[torch.arange(p.shape[0]), yc] -= vc.float()
            p.mul_(vc[:, None])
            gh[s:s + chunk] = p.to(w.dtype) @ w
            gw.addmm_(p.t(), hc.float())
        ctx.save_for_backward(gh.div_(n), gw.div_(n).to(w.dtype))
        return total / n

    @staticmethod
    def backward(ctx, g):
        gh, gw = ctx.saved_tensors
        return gh.mul_(g), gw.mul_(g), None, None


def chunked_lm_loss(model, chunk=256):
    """Make a transformers causal LM's ``model(input_ids=, labels=).loss`` use ChunkedLMLoss.

    ``lm_head`` is still called as a module (once per step), so DeepSpeed ZeRO-3 gathers
    and frees its weight as before. Returns the model.
    """
    head = model.lm_head

    def head_loss(self, h, y):
        return ChunkedLMLoss.apply(h.reshape(-1, h.shape[-1]), self.weight, y.reshape(-1), chunk)

    def forward(self, input_ids=None, labels=None, **kw):
        h = self.base_model(input_ids=input_ids).last_hidden_state
        y = torch.cat([labels[:, 1:], labels.new_full((labels.shape[0], 1), -100)], 1)  # next token
        return types.SimpleNamespace(loss=self.lm_head(h, y))

    head.forward = types.MethodType(head_loss, head)
    model.forward = types.MethodType(forward, model)
    return model
