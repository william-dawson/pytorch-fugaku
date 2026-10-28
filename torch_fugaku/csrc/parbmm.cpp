// CPU batched matmul that runs the batch in parallel. Without MKL, stock CPU
// bmm multiplies one slice at a time through a threaded BLAS, so a batch of
// small slices pays a full thread fork per slice. Here each thread takes a
// range of slices and multiplies them with mm; nested inside parallel_for,
// mm and the BLAS under it run single-threaded.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/ops/mm_cpu_dispatch.h>
#include <torch/library.h>

namespace {

at::Tensor parbmm(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "bmm expects 3-D tensors");
  TORCH_CHECK(a.size(0) == b.size(0) && a.size(2) == b.size(1),
              "bmm shape mismatch");
  const int64_t nb = a.size(0), m = a.size(1), k = a.size(2), n = b.size(2);
  auto out = at::empty({nb, m, n}, a.options());
  // Big slices keep the threaded BLAS: below ~16 MFLOP per slice, or with at
  // least as many slices as threads, the batch is the better axis.
  const bool by_batch = 2 * m * n * k < 16000000 || nb >= at::get_num_threads();
  if (by_batch) {
    at::parallel_for(0, nb, 1, [&](int64_t lo, int64_t hi) {
      // OpenMP workers do not inherit the caller's below-autograd dispatch
      // state; call the CPU kernel directly under an explicit guard.
      at::AutoDispatchBelowAutograd guard;
      for (int64_t i = lo; i < hi; ++i) {
        auto o = out.select(0, i);
        at::cpu::mm_out(o, a.select(0, i), b.select(0, i));
      }
    });
  } else {
    for (int64_t i = 0; i < nb; ++i) {
      auto o = out.select(0, i);
      at::cpu::mm_out(o, a.select(0, i), b.select(0, i));
    }
  }
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, CPU, m) { m.impl("bmm", parbmm); }
