#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <curand_kernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_normal_native.h>
#endif

namespace at::native {

namespace {

template <typename scalar_t, int N>
struct alignas(sizeof(scalar_t) * N) AlignedVec {
  scalar_t val[N];
};

// Scalar generate with bounds check, used for boundary elements.
template <typename scalar_t>
__device__ void normal_generate(
    scalar_t* output, int64_t base, int64_t elem, int64_t elem_end,
    curandStatePhilox4_32_10_t* state, double mean, double stddev) {
  float fmean = static_cast<float>(mean);
  float fstd = static_cast<float>(stddev);
  float4 n = curand_normal4(state);
  float vals[4] = {
    fmean + fstd * n.x, fmean + fstd * n.y,
    fmean + fstd * n.z, fmean + fstd * n.w
  };
  #pragma unroll
  for (int j = 0; j < 4 && elem + j < elem_end; j++) {
    output[base + elem + j] = static_cast<scalar_t>(vals[j]);
  }
}
template <>
__device__ void normal_generate<double>(
    double* output, int64_t base, int64_t elem, int64_t elem_end,
    curandStatePhilox4_32_10_t* state, double mean, double stddev) {
  double2 n = curand_normal2_double(state);
  output[base + elem] = mean + stddev * n.x;
  if (elem + 1 < elem_end) {
    output[base + elem + 1] = mean + stddev * n.y;
  }
}

// Generate one batch, skip first `skip` elements to align Box-Muller pairing
// to consistent 4-Philox-output group boundaries.
template <typename scalar_t>
__device__ void normal_generate_skip(
    scalar_t* output, int64_t base, int64_t elem, int64_t elem_end,
    curandStatePhilox4_32_10_t* state, double mean, double stddev, int skip) {
  float fmean = static_cast<float>(mean);
  float fstd = static_cast<float>(stddev);
  float4 n = curand_normal4(state);
  float vals[4] = {
    fmean + fstd * n.x, fmean + fstd * n.y,
    fmean + fstd * n.z, fmean + fstd * n.w
  };
  #pragma unroll
  for (int j = skip; j < 4 && elem + j - skip < elem_end; j++) {
    output[base + elem + j - skip] = static_cast<scalar_t>(vals[j]);
  }
}
template <>
__device__ void normal_generate_skip<double>(
    double* output, int64_t base, int64_t elem, int64_t elem_end,
    curandStatePhilox4_32_10_t* state, double mean, double stddev, int skip) {
  double2 n = curand_normal2_double(state);
  if (skip == 0) {
    output[base + elem] = mean + stddev * n.x;
    if (elem + 1 < elem_end) {
      output[base + elem + 1] = mean + stddev * n.y;
    }
  } else {
    // skip == 1: discard first value, write second.
    if (elem < elem_end) {
      output[base + elem] = mean + stddev * n.y;
    }
  }
}

// Vectorized generate without bounds check, uses aligned vector store.
template <typename scalar_t>
__device__ void normal_generate_vec(
    scalar_t* output, int64_t pos,
    curandStatePhilox4_32_10_t* state, double mean, double stddev) {
  float fmean = static_cast<float>(mean);
  float fstd = static_cast<float>(stddev);
  float4 n = curand_normal4(state);
  AlignedVec<scalar_t, 4> v;
  v.val[0] = static_cast<scalar_t>(fmean + fstd * n.x);
  v.val[1] = static_cast<scalar_t>(fmean + fstd * n.y);
  v.val[2] = static_cast<scalar_t>(fmean + fstd * n.z);
  v.val[3] = static_cast<scalar_t>(fmean + fstd * n.w);
  *reinterpret_cast<AlignedVec<scalar_t, 4>*>(&output[pos]) = v;
}

template <>
__device__ void normal_generate_vec<double>(
    double* output, int64_t pos,
    curandStatePhilox4_32_10_t* state, double mean, double stddev) {
  double2 n = curand_normal2_double(state);
  AlignedVec<double, 2> v;
  v.val[0] = mean + stddev * n.x;
  v.val[1] = mean + stddev * n.y;
  *reinterpret_cast<AlignedVec<double, 2>*>(&output[pos]) = v;
}

template <typename scalar_t, bool single_key, typename key_offset_calc_t>
__global__ void philox_normal_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ keys,
    int64_t num_keys,
    int64_t event_numel,
    int64_t elems_per_thread,
    double mean,
    double stddev,
    key_offset_calc_t key_offset_calc) {
  constexpr size_t compute_size =
      sizeof(scalar_t) < sizeof(float) ? sizeof(float) : sizeof(scalar_t);
  constexpr int outputs_per_normal = compute_size / sizeof(float);
  constexpr int elems_per_call = 4 / outputs_per_normal;

  int64_t num_chunks = (event_numel + elems_per_thread - 1) / elems_per_thread;
  int64_t total_threads = single_key ? num_chunks : num_keys * num_chunks;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  uint64_t seed, key_offset;
  if constexpr (single_key) {
    seed = keys[0];
    key_offset = keys[1];
  }

  for (; tid < total_threads; tid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t key_idx, chunk_idx;
    if constexpr (single_key) {
      key_idx = 0;
      chunk_idx = tid;
    } else {
      key_idx = tid / num_chunks;
      chunk_idx = tid % num_chunks;
      auto key_elem_offset = key_offset_calc.get(key_idx)[0];
      seed = keys[key_elem_offset];
      key_offset = keys[key_elem_offset + 1];
    }
    int64_t elem_start = chunk_idx * elems_per_thread;
    int64_t elem_end = min(elem_start + elems_per_thread, event_numel);
    int64_t base = single_key ? 0 : key_idx * event_numel;

    // Align curand init to a 4-Philox-output boundary so that Box-Muller
    // always pairs the same absolute stream positions, regardless of
    // key_offset parity.  Only possible when the misalignment is a whole
    // number of output elements (always true for float; true for double
    // when key_offset is even).
    int misalign = static_cast<int>(key_offset & 3);
    int skip = 0;
    unsigned long long philox_offset = key_offset +
        static_cast<unsigned long long>(elem_start) * outputs_per_normal;
    if (misalign > 0 && (misalign % outputs_per_normal) == 0) {
      skip = misalign / outputs_per_normal;
      philox_offset -= misalign;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence=*/0, /*offset=*/philox_offset, &state);

    int64_t elem = elem_start;

    if (skip > 0 && elem < elem_end) {
      normal_generate_skip<scalar_t>(
          output, base, elem, elem_end, &state, mean, stddev, skip);
      elem += min(static_cast<int64_t>(elems_per_call - skip),
                  elem_end - elem);
    }

    int64_t full_end = elem + ((elem_end - elem) / elems_per_call) * elems_per_call;
    // Vec stores need aligned positions; only possible without prefix skip.
    if (skip == 0 && (single_key || (base % elems_per_call) == 0)) {
      for (; elem < full_end; elem += elems_per_call) {
        normal_generate_vec<scalar_t>(output, base + elem, &state, mean, stddev);
      }
    } else {
      for (; elem < full_end; elem += elems_per_call) {
        normal_generate<scalar_t>(output, base, elem, elem_end, &state, mean, stddev);
      }
    }
    if (elem < elem_end) {
      normal_generate<scalar_t>(output, base, elem, elem_end, &state, mean, stddev);
    }
  }
}

} // anonymous namespace

Tensor& _philox_normal_cuda_(Tensor& self, const Tensor& key, double mean, double stddev) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_normal: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_normal: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(key.is_cuda(),
      "_philox_normal: key must be a CUDA tensor");
  TORCH_CHECK(self.is_cuda(),
      "_philox_normal: self must be a CUDA tensor");
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

  at::cuda::CUDAGuard device_guard(key.device());

  // Expand key batch dims to match self (lazy, no allocation).
  std::vector<int64_t> expanded_key_sizes;
  expanded_key_sizes.reserve(key_batch_ndim + 1);
  for (int64_t i = 0; i < key_batch_ndim; i++) {
    expanded_key_sizes.push_back(self.size(i));
  }
  expanded_key_sizes.push_back(2);
  auto key_expanded = key.expand(expanded_key_sizes);

  int64_t num_keys = key_expanded.numel() / 2;
  int64_t event_numel = self.numel() / num_keys;

  if (num_keys == 0 || event_numel == 0) {
    return self;
  }

  // Build an OffsetCalculator over the batch dims so the kernel can map
  // a linear key_idx to the correct element offset in the strided key tensor.
  // Strides are in elements (uint64_t), not bytes.
  // OffsetCalculator decomposes linear indices in column-major order (dim 0
  // is fastest), but our key indices are row-major.  Reverse the dims.
  std::vector<int64_t> batch_sizes(key_batch_ndim);
  std::vector<int64_t> batch_strides(key_batch_ndim);
  for (int64_t i = 0; i < key_batch_ndim; i++) {
    batch_sizes[i] = key_expanded.size(key_batch_ndim - 1 - i);
    batch_strides[i] = key_expanded.stride(key_batch_ndim - 1 - i);
  }
  const int64_t* batch_strides_ptr = batch_strides.data();
  auto key_offset_calc = OffsetCalculator<1>(
      key_batch_ndim, batch_sizes.data(), &batch_strides_ptr);

  constexpr int64_t elems_per_thread = 16;
  int64_t num_chunks = (event_numel + elems_per_thread - 1) / elems_per_thread;
  int64_t total_threads = num_keys * num_chunks;
  constexpr int block_size = 256;
  int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  int num_blocks = std::min(
      static_cast<int>((total_threads + block_size - 1) / block_size),
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm);

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "_philox_normal_cuda", [&] {
    if (num_keys == 1) {
      philox_normal_kernel<scalar_t, true><<<num_blocks, block_size, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          self.mutable_data_ptr<scalar_t>(),
          key_expanded.data_ptr<uint64_t>(),
          num_keys, event_numel, elems_per_thread, mean, stddev,
          key_offset_calc);
    } else {
      philox_normal_kernel<scalar_t, false><<<num_blocks, block_size, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          self.mutable_data_ptr<scalar_t>(),
          key_expanded.data_ptr<uint64_t>(),
          num_keys, event_numel, elems_per_thread, mean, stddev,
          key_offset_calc);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return self;
}

} // namespace at::native
