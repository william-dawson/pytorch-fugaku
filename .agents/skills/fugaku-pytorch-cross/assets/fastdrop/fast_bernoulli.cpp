// Fast CPU bernoulli_(p) for dropout masks on A64FX.
//
// Stock PyTorch's CPU bernoulli_ draws scalar random numbers element by element; on A64FX it
// was 71% of a GPT-125M training step (22.8 s of 32.1 s; Fujitsu's PyTorch 1.13 needs 0.69 s for
// the same 73 calls). This replaces aten::bernoulli_.float on CPU with a counter-based generator:
// element i of a call gets hash(i, key), where key is one 64-bit draw from torch's generator, so
// the loop has no carried state, vectorizes (SVE) and splits across threads, and
// torch.manual_seed still makes runs reproducible. The stream differs from stock PyTorch's.
//
// Load with torch.ops.load_library(<this .so>); registration happens at load.
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

// out[i] = 1 with probability p, else 0. Two keyed hash rounds decorrelate neighbouring indices
// and calls; the 32-bit result is compared against p * 2^32 in 64 bits so p = 1 gives all ones.
void fill_mask(float *out, int64_t n, uint64_t key, double p) {
  const uint32_t k0 = static_cast<uint32_t>(key), k1 = static_cast<uint32_t>(key >> 32);
  const uint64_t threshold = static_cast<uint64_t>(p * 4294967296.0);
  at::parallel_for(0, n, 1 << 15, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const uint32_t lo = static_cast<uint32_t>(i), hi = static_cast<uint32_t>(static_cast<uint64_t>(i) >> 32);
      const uint32_t x = mix32(mix32(lo ^ k0) + (k1 ^ hi));
      out[i] = static_cast<uint64_t>(x) < threshold ? 1.0f : 0.0f;
    }
  });
}

uint64_t draw_key(const std::optional<at::Generator> &gen) {
  auto *g = at::get_generator_or_default<at::CPUGeneratorImpl>(gen, at::detail::getDefaultCPUGenerator());
  std::lock_guard<std::mutex> lock(g->mutex_);
  return g->random64();
}

at::Tensor &bernoulli_float_(at::Tensor &self, double p, std::optional<at::Generator> gen) {
  TORCH_CHECK(0.0 <= p && p <= 1.0, "bernoulli_ expects p in [0, 1], got ", p);
  const uint64_t key = draw_key(gen);
  if (self.scalar_type() == at::kFloat && self.is_contiguous()) {
    fill_mask(self.data_ptr<float>(), self.numel(), key, p);
    return self;
  }
  at::Tensor mask = at::empty(self.sizes(), self.options().dtype(at::kFloat).memory_format(at::MemoryFormat::Contiguous));
  fill_mask(mask.data_ptr<float>(), mask.numel(), key, p);
  self.copy_(mask);
  return self;
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("bernoulli_.float", TORCH_FN(bernoulli_float_));
}
