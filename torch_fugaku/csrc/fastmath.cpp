// SVE kernels for the elementwise and reduction ops the stock aarch64 wheel runs
// at near-scalar speed on A64FX, registered as CPU overrides: gelu and
// gelu_backward (erf form), tanh, and the whole-tensor 2-norm that gradient
// clipping computes. tanh_backward already runs at memory speed and stays
// stock. Fast path: contiguous float32; everything else goes to the stock CPU
// kernel.
//
// tanhf, erff and the FEXPA-based expf are from Arm Optimized Routines
// (math/aarch64/sve), Copyright (c) 2019-2026 Arm Limited,
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception.
// Their stated maximum errors: tanhf 2.07 ULP, erff 1.93 ULP.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/gelu_backward_cpu_dispatch.h>
#include <ATen/ops/gelu_cpu_dispatch.h>
#include <ATen/ops/linalg_vector_norm_cpu_dispatch.h>
#include <ATen/ops/tanh_cpu_dispatch.h>
#include <arm_sve.h>
#include <cmath>
#include <functional>
#include <torch/library.h>

#include "aor_erff_table.h"

namespace {

// ---- Arm Optimized Routines, adapted (fixed ptrue predicates inlined) ----

// e^(2x) - 1, accurate for small x (AOR tanhf.c: e2xm1f_inline).
inline svfloat32_t e2xm1f(svfloat32_t x, svbool_t pg) {
  const svfloat32_t c0 = svdup_f32(0x1.fffffep-2f), c1 = svdup_f32(0x1.5554aep-3f);
  const svfloat32_t c2 = svdup_f32(0x1.555736p-5f), c3 = svdup_f32(0x1.12287cp-7f);
  const svfloat32_t c4 = svdup_f32(0x1.6b55a2p-10f);
  svfloat32_t j = svrinta_x(pg, svmul_x(pg, x, 0x1.715476p+1f));
  svfloat32_t f = svadd_x(pg, x, x);
  f = svmls_x(pg, f, j, 0x1.62e4p-1f);
  f = svmls_x(pg, f, j, 0x1.7f7d1cp-20f);
  svfloat32_t p12 = svmla_x(pg, c1, f, c2);
  svfloat32_t p34 = svmla_x(pg, c3, f, c4);
  svfloat32_t f2 = svmul_x(pg, f, f);
  svfloat32_t p = svmla_x(pg, p12, f2, p34);
  p = svmla_x(pg, c0, f, p);
  p = svmla_x(pg, f, f2, p);
  svfloat32_t t = svscale_x(pg, svdup_f32(1.0f), svcvt_s32_x(pg, j));
  return svmla_x(pg, svsub_x(pg, t, 1.0f), p, t);
}

// tanh(x) = (e^2x - 1) / (e^2x + 1); |x| > ~9.01 rounds to +-1 (AOR tanhf.c).
inline svfloat32_t sv_tanhf(svfloat32_t x, svbool_t pg) {
  svfloat32_t q = e2xm1f(x, pg);
  svfloat32_t y = svdiv_x(pg, q, svadd_x(pg, q, 2.0f));
  svbool_t big = svacgt(pg, x, 0x1.205966p+3f);
  svuint32_t sign = svand_x(pg, svreinterpret_u32(x), 0x80000000u);
  svfloat32_t one = svreinterpret_f32(svorr_x(pg, sign, svdup_u32(0x3f800000u)));
  return svsel(big, one, y);
}

// erf(x) from a 1/128-spaced table and a cubic correction (AOR erff.c).
inline svfloat32_t sv_erff(svfloat32_t x, svbool_t pg) {
  const float shift = 0x1p16f;
  svbool_t a_gt_min = svacgt(pg, x, 0x1.cp-7f);
  svbool_t a_ge_max = svacge(pg, x, 3.9375f);
  // Clamped so the table index stays within its 513 entries; lanes at or
  // above the bound are replaced by 1 below.
  svfloat32_t a = svmin_x(pg, svabs_x(pg, x), 3.9375f);
  svfloat32_t z = svadd_x(pg, a, shift);
  svuint32_t i = svand_x(pg, svreinterpret_u32(z), 0xfffu);
  i = svadd_x(pg, i, i);
  svfloat32_t r = svsub_z(a_gt_min, z, shift);
  svfloat32_t erfr = svld1_gather_index(a_gt_min, &aor_erff_tab[0].erf, i);
  svfloat32_t scale = svld1_gather_index(a_gt_min, &aor_erff_tab[0].scale, i);
  scale = svsel(a_gt_min, scale, svdup_f32(0x1.20dd76p+0f));
  svfloat32_t d = svsub_x(pg, a, r);
  svfloat32_t d2 = svmul_x(pg, d, d);
  svfloat32_t y = svmla_x(pg, r, d, 0x1.555556p-2f);
  y = svmla_x(pg, erfr, scale, svmls_x(pg, d, d2, y));
  y = svsel(a_ge_max, svdup_f32(1.0f), y);
  svuint32_t sign = svand_x(pg, svreinterpret_u32(x), 0x80000000u);
  return svreinterpret_f32(svorr_x(pg, sign, svreinterpret_u32(y)));
}

// e^x for x >= -87 via FEXPA (AOR expf.c fast path); callers clamp.
inline svfloat32_t sv_expf(svfloat32_t x, svbool_t pg) {
  const float sh = 0x1.803f8p17f;
  svfloat32_t z = svmad_x(pg, svdup_f32(0x1.715476p+0f), x, sh);
  svfloat32_t n = svsub_x(pg, z, sh);
  svfloat32_t r = svmls_x(pg, x, n, 0x1.62e4p-1f);
  r = svmls_x(pg, r, n, 0x1.7f7d1cp-20f);
  svfloat32_t scale = svexpa(svreinterpret_u32(z));
  svfloat32_t poly = svmla_x(pg, r, svmul_x(pg, r, r), 0.5f);
  return svmla_x(pg, scale, scale, poly);
}

// ---- elementwise drivers ----

bool fast(const at::Tensor& t) {
  return t.scalar_type() == at::kFloat && t.is_contiguous() && t.device().is_cpu();
}

template <typename F>
void each(int64_t n, F body) {
  at::parallel_for(0, n, at::internal::GRAIN_SIZE, [&](int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i += svcntw()) body(i, svwhilelt_b32(i, hi));
  });
}

constexpr float kInvSqrt2 = 0x1.6a09e6p-1f;   // 1/sqrt(2)
constexpr float kInvSqrt2Pi = 0x1.988454p-2f; // 1/sqrt(2*pi)

at::Tensor fm_gelu(const at::Tensor& self, c10::string_view approximate) {
  if (approximate != "none" || !fast(self)) return at::cpu::gelu(self, approximate);
  auto out = at::empty_like(self);
  const float* x = self.const_data_ptr<float>();
  float* y = out.mutable_data_ptr<float>();
  each(self.numel(), [&](int64_t i, svbool_t pg) {
    svfloat32_t v = svld1(pg, x + i);
    svfloat32_t e = sv_erff(svmul_x(pg, v, kInvSqrt2), pg);
    svst1(pg, y + i, svmul_x(pg, svmul_x(pg, v, 0.5f), svadd_x(pg, e, 1.0f)));
  });
  return out;
}

at::Tensor fm_gelu_backward(const at::Tensor& grad, const at::Tensor& self,
                         c10::string_view approximate) {
  if (approximate != "none" || !fast(self) || !fast(grad) || grad.sizes() != self.sizes())
    return at::cpu::gelu_backward(grad, self, approximate);
  auto out = at::empty_like(self);
  const float* g = grad.const_data_ptr<float>();
  const float* x = self.const_data_ptr<float>();
  float* y = out.mutable_data_ptr<float>();
  // d/dx gelu = 0.5 (1 + erf(x/sqrt2)) + x exp(-x^2/2) / sqrt(2 pi)
  each(self.numel(), [&](int64_t i, svbool_t pg) {
    svfloat32_t v = svld1(pg, x + i);
    svfloat32_t cdf = svmul_x(pg, svadd_x(pg, sv_erff(svmul_x(pg, v, kInvSqrt2), pg), 1.0f), 0.5f);
    // Below -87 FEXPA is invalid and e^arg is negligible: use exactly 0, so
    // x = +-inf gives inf * 0 = NaN as the stock kernel does.
    svfloat32_t arg = svmul_x(pg, svmul_x(pg, v, v), -0.5f);
    svbool_t tiny = svcmplt(pg, arg, -87.0f);
    svfloat32_t e = sv_expf(svmax_x(pg, arg, -87.0f), pg);
    svfloat32_t pdf = svmul_x(pg, svsel(tiny, svdup_f32(0.0f), e), kInvSqrt2Pi);
    svst1(pg, y + i, svmul_x(pg, svld1(pg, g + i), svmla_x(pg, cdf, v, pdf)));
  });
  return out;
}

at::Tensor fm_tanh(const at::Tensor& self) {
  if (!fast(self)) return at::cpu::tanh(self);
  auto out = at::empty_like(self);
  const float* x = self.const_data_ptr<float>();
  float* y = out.mutable_data_ptr<float>();
  each(self.numel(), [&](int64_t i, svbool_t pg) {
    svst1(pg, y + i, sv_tanhf(svld1(pg, x + i), pg));
  });
  return out;
}

// Sum of squares in double: no overflow, and a relative error of ~3e-8 on 10M values where
// the stock kernel (float32 accumulation) reaches 1e-3. Memory-bound; stock runs ~10x slower.
double sumsq(const float* x, int64_t lo, int64_t hi) {
  double acc = 0;
#pragma omp simd reduction(+ : acc)
  for (int64_t i = lo; i < hi; ++i) {
    const double v = x[i];
    acc += v * v;
  }
  return acc;
}

// The 2-norm of a whole contiguous float32 tensor (gradient clipping's per-tensor norm);
// any other ord, dim, keepdim or dtype takes the stock kernel.
at::Tensor fm_vector_norm(const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim,
                          bool keepdim, std::optional<at::ScalarType> dtype) {
  if (!fast(self) || dim.has_value() || keepdim || self.numel() == 0 || ord.isComplex() ||
      ord.toDouble() != 2.0 || (dtype.has_value() && *dtype != at::kFloat))
    return at::cpu::linalg_vector_norm(self, ord, dim, keepdim, dtype);
  const float* x = self.const_data_ptr<float>();
  const double s = at::parallel_reduce(
      0, self.numel(), at::internal::GRAIN_SIZE, 0.0,
      [&](int64_t lo, int64_t hi, double acc) { return acc + sumsq(x, lo, hi); }, std::plus<double>());
  return at::scalar_tensor(static_cast<float>(std::sqrt(s)), self.options());
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("linalg_vector_norm", fm_vector_norm);
  m.impl("gelu", fm_gelu);
  m.impl("gelu_backward", fm_gelu_backward);
  m.impl("tanh", fm_tanh);
}
