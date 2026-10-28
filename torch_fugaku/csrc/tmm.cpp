// CPU matrix multiply with a transposed right operand (X . W^T, nn.Linear's forward) via a fast
// transpose. On A64FX NVPL's transposed-B GEMM runs well below its plain one (down_proj-shaped
// 2048x8192 . (2048x8192)^T: 170 vs 800 GF/s), and torch's own transpose copy is too slow to
// help. Here a threaded, cache-blocked transpose fills a reused per-thread buffer, then the stock
// kernel multiplies untransposed operands. Fast path: float32, B a transposed view, B at most
// 128 MB; everything else takes the stock kernel.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/addmm_cpu_dispatch.h>
#include <ATen/ops/mm_cpu_dispatch.h>
#include <torch/library.h>

#include <vector>

namespace {

constexpr int64_t kMinElems = 1 << 16;      // below this the transpose cannot pay
constexpr int64_t kMaxBytes = 128LL << 20;  // larger weights stay stock: memory per rank is tight
constexpr int64_t kTile = 64;

// B is [K, N] with B.t() contiguous; true when it should be transposed first.
bool fast(const at::Tensor& a, const at::Tensor& b) {
  return a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat && a.device().is_cpu() &&
         b.device().is_cpu() && a.dim() == 2 && b.dim() == 2 && !b.is_contiguous() && b.t().is_contiguous() &&
         b.numel() >= kMinElems && b.numel() * 4 <= kMaxBytes;
}

// dst [K, N] row-major from src = B^T stored [N, K] row-major, in kTile x kTile blocks.
void transpose(const float* src, float* dst, int64_t K, int64_t N) {
  const int64_t kb = (K + kTile - 1) / kTile;
  at::parallel_for(0, kb, 1, [&](int64_t lo, int64_t hi) {
    for (int64_t bk = lo; bk < hi; ++bk) {
      const int64_t k0 = bk * kTile, k1 = std::min(K, k0 + kTile);
      for (int64_t n0 = 0; n0 < N; n0 += kTile) {
        const int64_t n1 = std::min(N, n0 + kTile);
        for (int64_t k = k0; k < k1; ++k)
          for (int64_t n = n0; n < n1; ++n) dst[k * N + n] = src[n * K + k];
      }
    }
  });
}

// The contiguous copy of B, in a per-thread buffer reused across calls.
at::Tensor contiguous_b(const at::Tensor& b) {
  thread_local std::vector<float> buf;
  const int64_t K = b.size(0), N = b.size(1);
  if (static_cast<int64_t>(buf.size()) < K * N) buf.resize(K * N);
  transpose(b.const_data_ptr<float>(), buf.data(), K, N);
  return at::from_blob(buf.data(), {K, N}, b.options());
}

at::Tensor tmm(const at::Tensor& a, const at::Tensor& b) {
  if (!fast(a, b)) return at::cpu::mm(a, b);
  return at::cpu::mm(a, contiguous_b(b));
}

at::Tensor taddmm(const at::Tensor& self, const at::Tensor& a, const at::Tensor& b, const at::Scalar& beta,
                  const at::Scalar& alpha) {
  if (!fast(a, b)) return at::cpu::addmm(self, a, b, beta, alpha);
  return at::cpu::addmm(self, a, contiguous_b(b), beta, alpha);
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("mm", tmm);
  m.impl("addmm", taddmm);
}
