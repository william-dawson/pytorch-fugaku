// CPU bernoulli_(p), the draw behind dropout's mask. The wheel's draws one
// scalar random number per element; on A64FX that was 71% of a GPT-125M
// training step (73 calls, 22.8 s of 32.1 s). Here element i of a call is a
// keyed integer hash of i: no state carried between elements, so the loop
// vectorizes and splits across threads. The key is one 64-bit draw from
// torch's generator per call, so torch.manual_seed still makes runs
// reproducible; the stream differs from the wheel's.
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <cstdint>
#include <mutex>
#include <optional>

namespace {

// "lowbias32" integer hash (Wellons): good avalanche at two multiplies.
inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// out[i] = 1 with probability p, else 0. Two keyed rounds decorrelate
// neighbouring indices and calls; comparing in 64 bits against p * 2^32 makes
// p = 1 all ones and p = 0 all zeros.
void fill_mask(float* out, int64_t n, uint64_t key, double p) {
  const uint32_t k0 = static_cast<uint32_t>(key), k1 = static_cast<uint32_t>(key >> 32);
  const uint64_t threshold = static_cast<uint64_t>(p * 4294967296.0);
  at::parallel_for(0, n, 1 << 15, [&](int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) {
      const uint32_t l = static_cast<uint32_t>(i), h = static_cast<uint32_t>(static_cast<uint64_t>(i) >> 32);
      const uint32_t x = mix32(mix32(l ^ k0) + (k1 ^ h));
      out[i] = static_cast<uint64_t>(x) < threshold ? 1.0f : 0.0f;
    }
  });
}

at::Tensor& bernoulli_float_(at::Tensor& self, double p, std::optional<at::Generator> gen) {
  TORCH_CHECK(0.0 <= p && p <= 1.0, "bernoulli_ expects p in [0, 1], got ", p);
  uint64_t key;
  {
    auto* g = at::get_generator_or_default<at::CPUGeneratorImpl>(gen, at::detail::getDefaultCPUGenerator());
    std::lock_guard<std::mutex> lock(g->mutex_);
    key = g->random64();
  }
  if (self.scalar_type() == at::kFloat && self.is_contiguous()) {
    fill_mask(self.data_ptr<float>(), self.numel(), key, p);
    return self;
  }
  // Other dtypes and layouts: the same draw into a contiguous float32 mask.
  at::Tensor mask = at::empty(self.sizes(), self.options().dtype(at::kFloat).memory_format(at::MemoryFormat::Contiguous));
  fill_mask(mask.data_ptr<float>(), mask.numel(), key, p);
  self.copy_(mask);
  return self;
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, CPU, m) { m.impl("bernoulli_.float", TORCH_FN(bernoulli_float_)); }
