// CPU matrix multiply with a transposed right operand (X . W^T, nn.Linear's forward) via a fast
// transpose. On A64FX NVPL's transposed-B GEMM runs well below its plain one (down_proj-shaped
// 2048x8192 . (2048x8192)^T: 170 vs 800 GF/s), and torch's own transpose copy is too slow to
// help. Here a threaded, cache-blocked transpose fills a reused per-thread buffer, then the stock
// kernel multiplies untransposed operands. Fast path: float32, B a transposed view, B at most
// 128 MB; everything else takes the stock kernel.
//
// With TORCH_FUGAKU_FJBLAS=1 (pf-run's PF_FJBLAS=1, the process on Fujitsu's OpenMP runtime) float32
// GEMMs instead go to Fujitsu's threaded BLAS (libfjlapackexsve, ~1150 vs NVPL's ~980 GF/s per
// CMG), found at run time with dlopen: nothing of Fujitsu's is built or linked in.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/addmm_cpu_dispatch.h>
#include <ATen/ops/mm_cpu_dispatch.h>
#include <arm_neon.h>
#include <dlfcn.h>
#include <torch/library.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <string>
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

// 4x4 block in registers: four rows of src (contiguous along k) become four rows of dst
// (contiguous along n).
inline void transpose4(const float* s, int64_t K, float* d, int64_t N) {
  const float32x4x2_t t01 = vtrnq_f32(vld1q_f32(s), vld1q_f32(s + K));
  const float32x4x2_t t23 = vtrnq_f32(vld1q_f32(s + 2 * K), vld1q_f32(s + 3 * K));
  vst1q_f32(d, vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0])));
  vst1q_f32(d + N, vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1])));
  vst1q_f32(d + 2 * N, vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0])));
  vst1q_f32(d + 3 * N, vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1])));
}

// dst [K, N] row-major from src = B^T stored [N, K] row-major, in kTile x kTile blocks of 4x4
// register transposes; ragged edges element by element.
void transpose(const float* src, float* dst, int64_t K, int64_t N) {
  const int64_t kb = (K + kTile - 1) / kTile, nb = (N + kTile - 1) / kTile;
  at::parallel_for(0, kb * nb, 1, [&](int64_t lo, int64_t hi) {
    for (int64_t b = lo; b < hi; ++b) {
      const int64_t k0 = (b / nb) * kTile, k1 = std::min(K, k0 + kTile);
      const int64_t n0 = (b % nb) * kTile, n1 = std::min(N, n0 + kTile);
      const int64_t k4 = k0 + (k1 - k0) / 4 * 4, n4 = n0 + (n1 - n0) / 4 * 4;
      for (int64_t n = n0; n < n4; n += 4)
        for (int64_t k = k0; k < k4; k += 4) transpose4(src + n * K + k, K, dst + k * N + n, N);
      for (int64_t n = n0; n < n1; ++n)
        for (int64_t k = (n < n4 ? k4 : k0); k < k1; ++k) dst[k * N + n] = src[n * K + k];
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

// Fortran sgemm_ (column-major), resolved once from Fujitsu's BLAS when TORCH_FUGAKU_FJBLAS=1.
using sgemm_t = void (*)(const char*, const char*, const int*, const int*, const int*, const float*, const float*,
                         const int*, const float*, const int*, const float*, float*, const int*);
sgemm_t fj_sgemm() {
  static const sgemm_t fn = []() -> sgemm_t {
    const char* on = std::getenv("TORCH_FUGAKU_FJBLAS");
    if (!on || std::string(on) != "1") return nullptr;
    void* h = dlopen("libfjlapackexsve.so.1", RTLD_NOW | RTLD_GLOBAL);
    return h ? reinterpret_cast<sgemm_t>(dlsym(h, "sgemm_")) : nullptr;
  }();
  return fn;
}

// A row-major 2-D float32 view as a BLAS operand: 'N' if contiguous, 'T' if a transposed view.
bool blas_operand(const at::Tensor& x, char* trans, int* ld) {
  if (x.is_contiguous()) {
    *trans = 'N';
    *ld = static_cast<int>(std::max<int64_t>(1, x.size(1)));
    return true;
  }
  if (x.t().is_contiguous()) {
    *trans = 'T';
    *ld = static_cast<int>(std::max<int64_t>(1, x.size(0)));
    return true;
  }
  return false;
}

// out = beta * out + alpha * a @ b through Fujitsu's sgemm, as column-major out^T = b^T a^T.
// False when the operands do not fit the fast path; out must be contiguous [M, N].
bool fj_gemm(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, float alpha, float beta) {
  const sgemm_t sgemm = fj_sgemm();
  char ta, tb;
  int lda, ldb;
  if (!sgemm || a.scalar_type() != at::kFloat || b.scalar_type() != at::kFloat || !a.device().is_cpu() ||
      !b.device().is_cpu() || a.dim() != 2 || b.dim() != 2 || a.numel() == 0 || b.numel() == 0 ||
      a.size(0) * a.size(1) * b.size(1) < kMinElems * 16 || !blas_operand(a, &ta, &lda) ||
      !blas_operand(b, &tb, &ldb) || a.size(0) > INT32_MAX || b.size(1) > INT32_MAX || a.size(1) > INT32_MAX)
    return false;
  const int m = static_cast<int>(b.size(1)), n = static_cast<int>(a.size(0)), k = static_cast<int>(a.size(1));
  sgemm(&tb, &ta, &m, &n, &k, &alpha, b.const_data_ptr<float>(), &ldb, a.const_data_ptr<float>(), &lda, &beta,
        out.mutable_data_ptr<float>(), &m);
  return true;
}

at::Tensor tmm(const at::Tensor& a, const at::Tensor& b) {
  if (fj_sgemm() && a.dim() == 2 && b.dim() == 2) {
    at::Tensor out = at::empty({a.size(0), b.size(1)}, a.options());
    if (fj_gemm(a, b, out, 1.0f, 0.0f)) return out;
  }
  if (!fast(a, b)) return at::cpu::mm(a, b);
  return at::cpu::mm(a, contiguous_b(b));
}

at::Tensor taddmm(const at::Tensor& self, const at::Tensor& a, const at::Tensor& b, const at::Scalar& beta,
                  const at::Scalar& alpha) {
  if (fj_sgemm() && a.dim() == 2 && b.dim() == 2 && self.scalar_type() == at::kFloat && !beta.isComplex() &&
      !alpha.isComplex()) {
    const float be = beta.toFloat();
    at::Tensor out = be == 0.0f ? at::empty({a.size(0), b.size(1)}, a.options())
                                : self.expand({a.size(0), b.size(1)}).contiguous();
    if (fj_gemm(a, b, out, alpha.toFloat(), be)) return out;
  }
  if (!fast(a, b)) return at::cpu::addmm(self, a, b, beta, alpha);
  return at::cpu::addmm(self, a, contiguous_b(b), beta, alpha);
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("mm", tmm);
  m.impl("addmm", taddmm);
}
