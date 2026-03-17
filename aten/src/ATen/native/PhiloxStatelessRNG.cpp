#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_key_fold_in_native.h>
#include <ATen/ops/_philox_key_split_native.h>
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#include <cmath>

namespace at::native {

namespace {

// Constants matching curand's conversion formulas.
constexpr float CURAND_2POW32_INV = 2.3283064e-10f;
constexpr double CURAND_2POW32_INV_DOUBLE = 2.3283064365386963e-10;
constexpr float CURAND_2POW32_INV_2PI = 2.3283064e-10f * 6.2831855f;
constexpr double CURAND_2POW53_INV_DOUBLE = 1.1102230246251565e-16;
constexpr double CURAND_PI_DOUBLE = 3.1415926535897932;

// curand's offset counts individual uint32 outputs; philox_engine's
// constructor offset counts 128-bit blocks (groups of 4 uint32).
inline philox_engine make_philox(uint64_t seed, uint64_t offset) {
  philox_engine engine(seed, /*subsequence=*/0, /*offset=*/offset / 4);
  for (uint64_t i = 0; i < offset % 4; i++) {
    engine();
  }
  return engine;
}

// Matches curand's _curand_uniform: uint32 -> float in (0, 1].
inline float philox_uniform_float(uint32_t x) {
  return x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
}

// Matches curand's _curand_uniform_double(unsigned int): uint32 -> double.
inline double philox_uniform_double(uint32_t x) {
  return x * CURAND_2POW32_INV_DOUBLE + CURAND_2POW32_INV_DOUBLE;
}

// Matches curand's _curand_box_muller: 2 uint32 -> 2 standard normal floats.
// Uses sinf/cosf matching curand's host-side code path.
inline std::pair<float, float> box_muller_float(uint32_t x, uint32_t y) {
  float u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
  float v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI / 2.0f);
  float s = std::sqrt(-2.0f * std::log(u));
  return {s * std::sin(v), s * std::cos(v)};
}

// Matches curand's _curand_box_muller_double: 4 uint32 -> 2 standard normal doubles.
inline std::pair<double, double> box_muller_double(
    uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1) {
  auto zx = static_cast<unsigned long long>(x0) ^
      (static_cast<unsigned long long>(x1) << (53 - 32));
  double u = zx * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE / 2.0);
  auto zy = static_cast<unsigned long long>(y0) ^
      (static_cast<unsigned long long>(y1) << (53 - 32));
  double v = zy * (CURAND_2POW53_INV_DOUBLE * 2.0) + CURAND_2POW53_INV_DOUBLE;
  double s = std::sqrt(-2.0 * std::log(u));
  return {s * std::sin(v * CURAND_PI_DOUBLE), s * std::cos(v * CURAND_PI_DOUBLE)};
}

// Match CUDA's elems_per_thread for cross-device consistency.
constexpr int64_t ELEMS_PER_CHUNK = 16;

// --------------- Uniform generation ---------------

template <typename scalar_t>
void uniform_fill(
    scalar_t* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset, double low, double high) {
  float flow = static_cast<float>(low);
  float frange = static_cast<float>(high - low);
  // outputs_per_value = 1 for float-sized types: no gaps between chunks.
  auto engine = make_philox(seed, key_offset);
  for (int64_t i = 0; i < numel; i++) {
    float u = philox_uniform_float(engine());
    output[base + i] = static_cast<scalar_t>(flow + frange * u);
  }
}

template <>
void uniform_fill<double>(
    double* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset, double low, double high) {
  double range = high - low;
  // outputs_per_value = 2 for double: chunks of ELEMS_PER_CHUNK with gaps
  // to match CUDA's offset formula (key_offset + elem_start * 2).
  for (int64_t chunk_start = 0; chunk_start < numel; chunk_start += ELEMS_PER_CHUNK) {
    int64_t chunk_end = std::min(chunk_start + ELEMS_PER_CHUNK, numel);
    uint64_t offset = key_offset +
        static_cast<uint64_t>(chunk_start) * 2;
    auto engine = make_philox(seed, offset);
    for (int64_t i = chunk_start; i < chunk_end; i++) {
      double u = philox_uniform_double(engine());
      output[base + i] = low + range * u;
    }
  }
}

// --------------- Normal generation ---------------

template <typename scalar_t>
void normal_fill(
    scalar_t* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset, double mean, double stddev) {
  float fmean = static_cast<float>(mean);
  float fstd = static_cast<float>(stddev);
  // outputs_per_normal = 1 for float-sized types: no gaps between chunks,
  // but need alignment for Box-Muller consistency.
  for (int64_t chunk_start = 0; chunk_start < numel; chunk_start += ELEMS_PER_CHUNK) {
    int64_t chunk_end = std::min(chunk_start + ELEMS_PER_CHUNK, numel);
    uint64_t philox_offset = key_offset +
        static_cast<uint64_t>(chunk_start);
    int misalign = static_cast<int>(philox_offset & 3);
    int skip = 0;
    if (misalign > 0) {
      skip = misalign;
      philox_offset -= misalign;
    }
    auto engine = make_philox(seed, philox_offset);
    int64_t elem = chunk_start;

    // First partial group (skip leading values).
    if (skip > 0 && elem < chunk_end) {
      uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
      auto [n0, n1] = box_muller_float(r0, r1);
      auto [n2, n3] = box_muller_float(r2, r3);
      float normals[4] = {n0, n1, n2, n3};
      for (int j = skip; j < 4 && elem < chunk_end; j++, elem++) {
        output[base + elem] = static_cast<scalar_t>(fmean + fstd * normals[j]);
      }
    }

    // Full groups of 4.
    for (; elem + 4 <= chunk_end; elem += 4) {
      uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
      auto [n0, n1] = box_muller_float(r0, r1);
      auto [n2, n3] = box_muller_float(r2, r3);
      output[base + elem + 0] = static_cast<scalar_t>(fmean + fstd * n0);
      output[base + elem + 1] = static_cast<scalar_t>(fmean + fstd * n1);
      output[base + elem + 2] = static_cast<scalar_t>(fmean + fstd * n2);
      output[base + elem + 3] = static_cast<scalar_t>(fmean + fstd * n3);
    }

    // Remaining elements.
    if (elem < chunk_end) {
      uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
      auto [n0, n1] = box_muller_float(r0, r1);
      auto [n2, n3] = box_muller_float(r2, r3);
      float normals[4] = {n0, n1, n2, n3};
      for (int j = 0; elem < chunk_end; j++, elem++) {
        output[base + elem] = static_cast<scalar_t>(fmean + fstd * normals[j]);
      }
    }
  }
}

template <>
void normal_fill<double>(
    double* output, int64_t base, int64_t numel,
    uint64_t seed, uint64_t key_offset, double mean, double stddev) {
  // outputs_per_normal = 2 for double.
  for (int64_t chunk_start = 0; chunk_start < numel; chunk_start += ELEMS_PER_CHUNK) {
    int64_t chunk_end = std::min(chunk_start + ELEMS_PER_CHUNK, numel);
    uint64_t philox_offset = key_offset +
        static_cast<uint64_t>(chunk_start) * 2;
    int misalign = static_cast<int>(philox_offset & 3);
    int skip = 0;
    if (misalign > 0 && (misalign % 2) == 0) {
      skip = misalign / 2;
      philox_offset -= misalign;
    }
    auto engine = make_philox(seed, philox_offset);
    int64_t elem = chunk_start;

    // First partial group (skip leading doubles).
    if (skip > 0 && elem < chunk_end) {
      uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
      auto [n0, n1] = box_muller_double(r0, r1, r2, r3);
      // skip == 1: discard n0, use n1.
      if (elem < chunk_end) {
        output[base + elem] = mean + stddev * n1;
        elem++;
      }
    }

    // Full groups: 4 uint32 -> 2 doubles.
    for (; elem + 2 <= chunk_end; elem += 2) {
      uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
      auto [n0, n1] = box_muller_double(r0, r1, r2, r3);
      output[base + elem + 0] = mean + stddev * n0;
      output[base + elem + 1] = mean + stddev * n1;
    }

    // Remaining element.
    if (elem < chunk_end) {
      uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
      auto [n0, n1] = box_muller_double(r0, r1, r2, r3);
      output[base + elem] = mean + stddev * n0;
      elem++;
    }
  }
}

} // anonymous namespace

Tensor _philox_key_split_cpu(const Tensor& key, int64_t num_splits) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_split: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_split: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(num_splits > 0,
      "_philox_key_split: num_splits must be positive, got ",
      num_splits);

  auto key_contig = key.contiguous();
  int64_t num_keys = key.numel() / 2;

  auto batch_sizes = key.sizes().slice(0, key.dim() - 1);
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(batch_sizes.size() + 2);
  output_sizes.push_back(num_splits);
  for (auto s : batch_sizes) {
    output_sizes.push_back(s);
  }
  output_sizes.push_back(2);

  Tensor output = at::empty(output_sizes, key.options());

  if (num_keys == 0) {
    return output;
  }

  const uint64_t* input = key_contig.const_data_ptr<uint64_t>();
  uint64_t* out_ptr = output.data_ptr<uint64_t>();

  for (int64_t key_idx = 0; key_idx < num_keys; key_idx++) {
    uint64_t seed = input[key_idx * 2];
    uint64_t offset = input[key_idx * 2 + 1];
    auto engine = make_philox(seed, offset);

    for (int64_t split_idx = 0; split_idx < num_splits; split_idx++) {
      uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
      uint64_t new_seed = static_cast<uint64_t>(r0) |
          (static_cast<uint64_t>(r1) << 32);
      uint64_t new_offset = static_cast<uint64_t>(r2) |
          (static_cast<uint64_t>(r3) << 32);
      out_ptr[(split_idx * num_keys + key_idx) * 2] = new_seed;
      out_ptr[(split_idx * num_keys + key_idx) * 2 + 1] = new_offset;
    }
  }

  return output;
}

Tensor _philox_key_fold_in_cpu(const Tensor& key, int64_t data) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_fold_in: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_fold_in: key must have dtype uint64, got ",
      key.scalar_type());

  auto key_contig = key.contiguous();
  int64_t num_keys = key.numel() / 2;

  Tensor output = at::empty_like(key_contig);

  if (num_keys == 0) {
    return output;
  }

  const uint64_t* input = key_contig.const_data_ptr<uint64_t>();
  uint64_t* out_ptr = output.data_ptr<uint64_t>();

  for (int64_t idx = 0; idx < num_keys; idx++) {
    uint64_t seed = input[idx * 2];
    uint64_t offset = input[idx * 2 + 1];

    // Match CUDA: curand_init(seed, 0, offset), skipahead(data * 4).
    auto engine = make_philox(seed, offset + static_cast<uint64_t>(data) * 4);

    uint32_t r0 = engine(), r1 = engine(), r2 = engine(), r3 = engine();
    uint64_t new_seed = static_cast<uint64_t>(r0) |
        (static_cast<uint64_t>(r1) << 32);
    uint64_t new_offset = static_cast<uint64_t>(r2) |
        (static_cast<uint64_t>(r3) << 32);
    out_ptr[idx * 2] = new_seed;
    out_ptr[idx * 2 + 1] = new_offset;
  }

  return output;
}

Tensor& _philox_uniform_cpu_(Tensor& self, const Tensor& key, double low, double high) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_uniform: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_uniform: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(self.is_floating_point(),
      "_philox_uniform: self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      "_philox_uniform: self and key must be on the same device, got ",
      self.device(), " and ", key.device());

  int64_t key_batch_ndim = key.dim() - 1;
  TORCH_CHECK(self.dim() >= key_batch_ndim,
      "_philox_uniform: self must have at least ", key_batch_ndim,
      " dimensions to match key batch dims, got ", self.dim());

  for (int64_t i = 0; i < key_batch_ndim; i++) {
    TORCH_CHECK(key.size(i) == 1 || key.size(i) == self.size(i),
        "_philox_uniform: key batch dim ", i, " has size ", key.size(i),
        " which is incompatible with self dim size ", self.size(i));
  }

  std::vector<int64_t> expanded_key_sizes;
  expanded_key_sizes.reserve(key_batch_ndim + 1);
  for (int64_t i = 0; i < key_batch_ndim; i++) {
    expanded_key_sizes.push_back(self.size(i));
  }
  expanded_key_sizes.push_back(2);
  auto key_expanded = key.expand(expanded_key_sizes).contiguous();

  int64_t num_keys = key_expanded.numel() / 2;
  int64_t event_numel = self.numel() / num_keys;

  if (num_keys == 0 || event_numel == 0) {
    return self;
  }

  const uint64_t* keys_ptr = key_expanded.const_data_ptr<uint64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_cpu", [&] {
    scalar_t* out_ptr = self.data_ptr<scalar_t>();
    for (int64_t key_idx = 0; key_idx < num_keys; key_idx++) {
      uint64_t seed = keys_ptr[key_idx * 2];
      uint64_t key_offset = keys_ptr[key_idx * 2 + 1];
      int64_t base = key_idx * event_numel;
      uniform_fill<scalar_t>(out_ptr, base, event_numel, seed, key_offset, low, high);
    }
  });

  return self;
}

Tensor& _philox_normal_cpu_(Tensor& self, const Tensor& key, double mean, double stddev) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_normal: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_normal: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(self.is_floating_point(),
      "_philox_normal: self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      "_philox_normal: self and key must be on the same device, got ",
      self.device(), " and ", key.device());

  int64_t key_batch_ndim = key.dim() - 1;
  TORCH_CHECK(self.dim() >= key_batch_ndim,
      "_philox_normal: self must have at least ", key_batch_ndim,
      " dimensions to match key batch dims, got ", self.dim());

  for (int64_t i = 0; i < key_batch_ndim; i++) {
    TORCH_CHECK(key.size(i) == 1 || key.size(i) == self.size(i),
        "_philox_normal: key batch dim ", i, " has size ", key.size(i),
        " which is incompatible with self dim size ", self.size(i));
  }

  std::vector<int64_t> expanded_key_sizes;
  expanded_key_sizes.reserve(key_batch_ndim + 1);
  for (int64_t i = 0; i < key_batch_ndim; i++) {
    expanded_key_sizes.push_back(self.size(i));
  }
  expanded_key_sizes.push_back(2);
  auto key_expanded = key.expand(expanded_key_sizes).contiguous();

  int64_t num_keys = key_expanded.numel() / 2;
  int64_t event_numel = self.numel() / num_keys;

  if (num_keys == 0 || event_numel == 0) {
    return self;
  }

  const uint64_t* keys_ptr = key_expanded.const_data_ptr<uint64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "_philox_normal_cpu", [&] {
    scalar_t* out_ptr = self.data_ptr<scalar_t>();
    for (int64_t key_idx = 0; key_idx < num_keys; key_idx++) {
      uint64_t seed = keys_ptr[key_idx * 2];
      uint64_t key_offset = keys_ptr[key_idx * 2 + 1];
      int64_t base = key_idx * event_numel;
      normal_fill<scalar_t>(out_ptr, base, event_numel, seed, key_offset, mean, stddev);
    }
  });

  return self;
}

} // namespace at::native
