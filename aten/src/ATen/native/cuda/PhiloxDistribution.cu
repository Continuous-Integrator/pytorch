#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <curand_kernel.h>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#endif

namespace at::native {

namespace {

template <typename scalar_t, int N>
struct alignas(sizeof(scalar_t) * N) AlignedVec {
  scalar_t val[N];
};

// -- Distribution policies --------------------------------------------------
//
// Each policy provides the curand calls and transforms for one distribution.
// Template parameter scalar_t selects float4 vs double2 generation at compile
// time via if-constexpr.  The kernel template is parameterised on the policy
// so that Box-Muller alignment logic compiles away for distributions that do
// not need it.

template <typename scalar_t>
struct UniformPolicy {
  static constexpr bool needs_alignment = false;
  static constexpr const char* name = "_philox_uniform";
  float flow, frange;
  double low, high;

  __host__ __device__ UniformPolicy(double low_, double high_)
      : flow(static_cast<float>(low_)),
        frange(static_cast<float>(high_ - low_)),
        low(low_), high(high_) {}

  __device__ void generate(
      scalar_t* output, int64_t base, int64_t elem, int64_t elem_end,
      curandStatePhilox4_32_10_t* state) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
      double range = high - low;
      double2 u = curand_uniform2_double(state);
      output[base + elem] = low + range * u.x;
      if (elem + 1 < elem_end) {
        output[base + elem + 1] = low + range * u.y;
      }
    } else {
      float4 u = curand_uniform4(state);
      float vals[4] = {
        flow + frange * u.x, flow + frange * u.y,
        flow + frange * u.z, flow + frange * u.w
      };
      #pragma unroll
      for (int j = 0; j < 4 && elem + j < elem_end; j++) {
        output[base + elem + j] = static_cast<scalar_t>(vals[j]);
      }
    }
  }

  __device__ void generate_vec(
      scalar_t* output, int64_t pos,
      curandStatePhilox4_32_10_t* state) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
      double range = high - low;
      double2 u = curand_uniform2_double(state);
      AlignedVec<double, 2> v;
      v.val[0] = low + range * u.x;
      v.val[1] = low + range * u.y;
      *reinterpret_cast<AlignedVec<double, 2>*>(&output[pos]) = v;
    } else {
      float4 u = curand_uniform4(state);
      AlignedVec<scalar_t, 4> v;
      v.val[0] = static_cast<scalar_t>(flow + frange * u.x);
      v.val[1] = static_cast<scalar_t>(flow + frange * u.y);
      v.val[2] = static_cast<scalar_t>(flow + frange * u.z);
      v.val[3] = static_cast<scalar_t>(flow + frange * u.w);
      *reinterpret_cast<AlignedVec<scalar_t, 4>*>(&output[pos]) = v;
    }
  }

  __device__ bool tiled_ok(uint64_t /*key_offset*/) const { return true; }

  __device__ void tiled_body(
      float* warp_smem, int lane, int PADDED, int k,
      curandStatePhilox4_32_10_t* state) const {
    float4 u = curand_uniform4(state);
    warp_smem[lane * PADDED + k * 4 + 0] = flow + frange * u.x;
    warp_smem[lane * PADDED + k * 4 + 1] = flow + frange * u.y;
    warp_smem[lane * PADDED + k * 4 + 2] = flow + frange * u.z;
    warp_smem[lane * PADDED + k * 4 + 3] = flow + frange * u.w;
  }
};

template <typename scalar_t>
struct NormalPolicy {
  static constexpr bool needs_alignment = true;
  static constexpr const char* name = "_philox_normal";
  float fmean, fstd;
  double mean, stddev;

  __host__ __device__ NormalPolicy(double mean_, double stddev_)
      : fmean(static_cast<float>(mean_)),
        fstd(static_cast<float>(stddev_)),
        mean(mean_), stddev(stddev_) {}

  // Align curand init to a 4-Philox-output boundary so that Box-Muller
  // always pairs the same absolute stream positions, regardless of
  // key_offset parity.
  __device__ static unsigned long long align_offset(
      uint64_t key_offset, int& skip, int outputs_per_value) {
    int misalign = static_cast<int>(key_offset & 3);
    skip = 0;
    unsigned long long aligned = key_offset;
    if (misalign > 0 && (misalign % outputs_per_value) == 0) {
      skip = misalign / outputs_per_value;
      aligned -= misalign;
    }
    return aligned;
  }

  __device__ void generate(
      scalar_t* output, int64_t base, int64_t elem, int64_t elem_end,
      curandStatePhilox4_32_10_t* state) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
      double2 n = curand_normal2_double(state);
      output[base + elem] = mean + stddev * n.x;
      if (elem + 1 < elem_end) {
        output[base + elem + 1] = mean + stddev * n.y;
      }
    } else {
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
  }

  // Generate with skip: discard the first `skip` elements of the curand
  // batch to re-align after a non-aligned key_offset.
  __device__ void generate_skip(
      scalar_t* output, int64_t base, int64_t elem, int64_t elem_end,
      curandStatePhilox4_32_10_t* state, int skip) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
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
    } else {
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
  }

  __device__ void generate_vec(
      scalar_t* output, int64_t pos,
      curandStatePhilox4_32_10_t* state) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
      double2 n = curand_normal2_double(state);
      AlignedVec<double, 2> v;
      v.val[0] = mean + stddev * n.x;
      v.val[1] = mean + stddev * n.y;
      *reinterpret_cast<AlignedVec<double, 2>*>(&output[pos]) = v;
    } else {
      float4 n = curand_normal4(state);
      AlignedVec<scalar_t, 4> v;
      v.val[0] = static_cast<scalar_t>(fmean + fstd * n.x);
      v.val[1] = static_cast<scalar_t>(fmean + fstd * n.y);
      v.val[2] = static_cast<scalar_t>(fmean + fstd * n.z);
      v.val[3] = static_cast<scalar_t>(fmean + fstd * n.w);
      *reinterpret_cast<AlignedVec<scalar_t, 4>*>(&output[pos]) = v;
    }
  }

  __device__ bool tiled_ok(uint64_t key_offset) const {
    return (key_offset & 3) == 0;
  }

  __device__ void tiled_body(
      float* warp_smem, int lane, int PADDED, int k,
      curandStatePhilox4_32_10_t* state) const {
    float4 n = curand_normal4(state);
    warp_smem[lane * PADDED + k * 4 + 0] = fmean + fstd * n.x;
    warp_smem[lane * PADDED + k * 4 + 1] = fmean + fstd * n.y;
    warp_smem[lane * PADDED + k * 4 + 2] = fmean + fstd * n.z;
    warp_smem[lane * PADDED + k * 4 + 3] = fmean + fstd * n.w;
  }
};

// -- Unified Philox distribution kernel -------------------------------------

template <typename scalar_t, bool single_key, typename DistPolicy, typename key_offset_calc_t>
__global__ void philox_distribution_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ keys,
    int64_t num_keys,
    int64_t event_numel,
    int64_t elems_per_thread,
    DistPolicy dist,
    key_offset_calc_t key_offset_calc) {
  constexpr size_t compute_size =
      sizeof(scalar_t) < sizeof(float) ? sizeof(float) : sizeof(scalar_t);
  constexpr int outputs_per_value = compute_size / sizeof(float);
  constexpr int elems_per_call = 4 / outputs_per_value;

  extern __shared__ char philox_smem_[];

  // Generate elements in [elem_start, elem_end) from a curand state,
  // handling alignment (for normal), vectorized stores, and 64-bit
  // offset wrap.
  auto generate_range = [&](scalar_t* out, int64_t base, int64_t elem_start,
                            int64_t elem_end, uint64_t seed,
                            uint64_t key_offset, bool use_vec) {
    unsigned long long raw_offset = key_offset +
        static_cast<unsigned long long>(elem_start) * outputs_per_value;
    unsigned long long philox_offset = raw_offset;
    int skip = 0;

    if constexpr (DistPolicy::needs_alignment) {
      unsigned long long aligned_base =
          DistPolicy::align_offset(key_offset, skip, outputs_per_value);
      philox_offset = aligned_base +
          static_cast<unsigned long long>(elem_start) * outputs_per_value;
    }

    // Detect if the 64-bit offset wraps within this thread's range.
    auto outputs_in_range =
        static_cast<unsigned long long>(elem_end - elem_start) * outputs_per_value;
    bool range_wraps = raw_offset != 0 &&
        (raw_offset + outputs_in_range < raw_offset);
    int64_t wrap_elem = range_wraps
        ? elem_start + static_cast<int64_t>(
              (0ULL - raw_offset) / outputs_per_value)
        : elem_end;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence=*/0, /*offset=*/philox_offset, &state);

    int64_t gen_end = min(wrap_elem, elem_end);
    int64_t elem = elem_start;

    if constexpr (DistPolicy::needs_alignment) {
      if (skip > 0 && elem < gen_end) {
        dist.generate_skip(out, base, elem, gen_end, &state, skip);
        elem += min(static_cast<int64_t>(elems_per_call - skip),
                    gen_end - elem);
      }
    }

    int64_t full_end = elem + ((gen_end - elem) / elems_per_call) * elems_per_call;
    if ((!DistPolicy::needs_alignment || skip == 0) && use_vec) {
      for (; elem < full_end; elem += elems_per_call) {
        dist.generate_vec(out, base + elem, &state);
      }
    } else {
      for (; elem < full_end; elem += elems_per_call) {
        dist.generate(out, base, elem, gen_end, &state);
      }
    }
    if (elem < gen_end) {
      dist.generate(out, base, elem, gen_end, &state);
    }

    if (range_wraps) {
      curand_init(seed, /*subsequence=*/0, /*offset=*/0ULL, &state);
      elem = wrap_elem;
      full_end = elem + ((elem_end - elem) / elems_per_call) * elems_per_call;
      for (; elem < full_end; elem += elems_per_call) {
        dist.generate(out, base, elem, elem_end, &state);
      }
      if (elem < elem_end) {
        dist.generate(out, base, elem, elem_end, &state);
      }
    }
  };

  if constexpr (single_key) {
    uint64_t seed = keys[0];
    uint64_t key_offset = keys[1];

    // For non-double types, use warp-cooperative tiled generation with
    // shared memory transpose for coalesced writes.  Each warp processes
    // 1024-element tiles: 32 threads x 8 curand calls x 4 elements/call.
    // Values go to shared memory in thread-major order, then are read
    // back in position-major order for coalesced stores.
    bool could_wrap = key_offset != 0 &&
        (key_offset + static_cast<unsigned long long>(event_numel) *
         outputs_per_value < key_offset);
    if constexpr (elems_per_call == 4) {
      if (dist.tiled_ok(key_offset) && !could_wrap) {
        constexpr int K = 8;
        constexpr int EPT = elems_per_call * K;  // 32 elements per thread
        constexpr int TILE = 32 * EPT;            // 1024 elements per tile
        constexpr int PADDED = EPT + 1;            // 33, bank-conflict-free

        float* smem = reinterpret_cast<float*>(philox_smem_);
        int warp_id = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        float* warp_smem = smem + warp_id * 32 * PADDED;

        int global_warp =
            (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
        int num_warps = static_cast<int>(gridDim.x) * (blockDim.x / 32);

        for (int64_t tile = static_cast<int64_t>(global_warp) * TILE;
             tile < event_numel;
             tile += static_cast<int64_t>(num_warps) * TILE) {

          int64_t my_start = tile + static_cast<int64_t>(lane) * EPT;
          unsigned long long philox_offset = key_offset +
              static_cast<unsigned long long>(my_start) * outputs_per_value;

          curandStatePhilox4_32_10_t state;
          curand_init(seed, /*subsequence=*/0, /*offset=*/philox_offset, &state);

          #pragma unroll
          for (int k = 0; k < K; k++) {
            dist.tiled_body(warp_smem, lane, PADDED, k, &state);
          }
          __syncwarp();

          // Read in position-major order: step m reads thread m's row,
          // lane selects the element.  Consecutive lanes write consecutive
          // global addresses -> coalesced stores.
          #pragma unroll
          for (int m = 0; m < EPT; m++) {
            int64_t pos = tile + static_cast<int64_t>(m) * 32 + lane;
            if (pos < event_numel) {
              output[pos] = static_cast<scalar_t>(warp_smem[m * PADDED + lane]);
            }
          }
          __syncwarp();
        }
        return;
      }
    }

    // Fallback: contiguous per thread.
    int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t per_thread =
        ((event_numel + total_threads - 1) / total_threads +
         elems_per_call - 1) / elems_per_call * elems_per_call;

    int64_t elem_start = tid * per_thread;
    if (elem_start >= event_numel) return;
    int64_t elem_end = min(elem_start + per_thread, event_numel);

    generate_range(output, 0, elem_start, elem_end, seed, key_offset, true);
    return;
  }

  // Multi-key: each thread handles a fixed-size chunk for one key.
  // curand_init is called per chunk.
  int64_t num_chunks = (event_numel + elems_per_thread - 1) / elems_per_thread;
  int64_t total_threads = num_keys * num_chunks;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (; tid < total_threads; tid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t key_idx = tid / num_chunks;
    int64_t chunk_idx = tid % num_chunks;
    auto key_elem_offset = key_offset_calc.get(key_idx)[0];
    uint64_t seed = keys[key_elem_offset];
    uint64_t key_offset = keys[key_elem_offset + 1];

    int64_t elem_start = chunk_idx * elems_per_thread;
    int64_t elem_end = min(elem_start + elems_per_thread, event_numel);
    int64_t base = key_idx * event_numel;

    generate_range(output, base, elem_start, elem_end, seed, key_offset,
                   (base % elems_per_call) == 0);
  }
}

// -- Host-side shared implementation ----------------------------------------

// Non-portable fast path: use PyTorch's distribution_nullary_kernel with
// PhiloxCudaState pointing directly at the key's device memory.  Only
// supports single (non-batched) keys.
template <template<typename> class DistPolicyT>
Tensor& philox_nonportable_impl(
    Tensor& self, const Tensor& key,
    double param1, double param2);

template <>
Tensor& philox_nonportable_impl<UniformPolicy>(
    Tensor& self, const Tensor& key,
    double low, double high) {
  at::cuda::CUDAGuard device_guard(key.device());

  PhiloxCudaState philox_state;
  philox_state.seed_.ptr = reinterpret_cast<int64_t*>(key.data_ptr<uint64_t>());
  philox_state.offset_.ptr = reinterpret_cast<int64_t*>(key.data_ptr<uint64_t>() + 1);
  philox_state.offset_intragraph_ = 0;
  philox_state.captured_ = true;

  auto iter = TensorIterator::borrowing_nullary_op(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_cuda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    auto range = static_cast<opmath_t>(high - low);
    auto from = static_cast<scalar_t>(low);
    auto to = static_cast<scalar_t>(high);
    if (std::is_same_v<scalar_t, double>) {
      distribution_nullary_kernel<scalar_t, opmath_t, double2>(
          iter, philox_state,
          [] __device__ (curandStatePhilox4_32_10_t* state) -> double2 {
            return curand_uniform2_double(state);
          },
          [range, from, to] __device__ (opmath_t rand) {
            auto value = static_cast<scalar_t>(rand * range + from);
            return value == to ? from : value;
          });
    } else {
      distribution_nullary_kernel<scalar_t, opmath_t, float4>(
          iter, philox_state,
          [] __device__ (curandStatePhilox4_32_10_t* state) -> float4 {
            return curand_uniform4(state);
          },
          [range, from, to] __device__ (opmath_t rand) {
            auto value = static_cast<scalar_t>(rand * range + from);
            return value == to ? from : value;
          });
    }
  });
  return self;
}

template <>
Tensor& philox_nonportable_impl<NormalPolicy>(
    Tensor& self, const Tensor& key,
    double mean, double stddev) {
  at::cuda::CUDAGuard device_guard(key.device());

  PhiloxCudaState philox_state;
  philox_state.seed_.ptr = reinterpret_cast<int64_t*>(key.data_ptr<uint64_t>());
  philox_state.offset_.ptr = reinterpret_cast<int64_t*>(key.data_ptr<uint64_t>() + 1);
  philox_state.offset_intragraph_ = 0;
  philox_state.captured_ = true;

  auto iter = TensorIterator::borrowing_nullary_op(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "_philox_normal_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto fmean = static_cast<accscalar_t>(mean);
    auto fstd = static_cast<accscalar_t>(stddev);
    if (std::is_same_v<scalar_t, double>) {
      distribution_nullary_kernel<scalar_t, accscalar_t, double2>(
          iter, philox_state,
          [] __device__ (curandStatePhilox4_32_10_t* state) -> double2 {
            return curand_normal2_double(state);
          },
          [fmean, fstd] __device__ (accscalar_t rand) {
            return static_cast<scalar_t>(rand * fstd + fmean);
          });
    } else {
      distribution_nullary_kernel<scalar_t, accscalar_t, float4>(
          iter, philox_state,
          [] __device__ (curandStatePhilox4_32_10_t* state) -> float4 {
            return curand_normal4(state);
          },
          [fmean, fstd] __device__ (accscalar_t rand) {
            return static_cast<scalar_t>(rand * fstd + fmean);
          });
    }
  });
  return self;
}

template <template<typename> class DistPolicyT>
Tensor& philox_distribution_impl(
    Tensor& self, const Tensor& key,
    double param1, double param2, bool portable) {
  constexpr const char* op_name = DistPolicyT<float>::name;
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      op_name, ": key must have shape (2,) or (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      op_name, ": key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(key.is_cuda(),
      op_name, ": key must be a CUDA tensor");
  TORCH_CHECK(self.is_cuda(),
      op_name, ": self must be a CUDA tensor");
  TORCH_CHECK(self.is_floating_point(),
      op_name, ": self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      op_name, ": self and key must be on the same device, got ",
      self.device(), " and ", key.device());

  if (!portable) {
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 2,
        op_name, ": portable=False does not support batched keys");
    return philox_nonportable_impl<DistPolicyT>(self, key, param1, param2);
  }

  if (self.numel() == 0) {
    return self;
  }

  at::cuda::CUDAGuard device_guard(key.device());

  int64_t ndim = self.dim();
  int64_t elems_per_key = 1;
  int64_t key_dims = 0;

  if (key.dim() > 1) {
    // Batched: key.dim() == self.dim() + 1, with right-aligned broadcasting.
    // The trailing contiguous suffix of size-1 key dims forms the sequential
    // generation axis; all preceding dims index keys.
    TORCH_CHECK(key.dim() == ndim + 1,
        op_name, ": batched key must have ndim == output ndim + 1, "
        "got key shape ", key.sizes(), " with output shape ", self.sizes());

    for (int64_t i = 0; i < ndim; i++) {
      TORCH_CHECK(key.size(i) == 1 || key.size(i) == self.size(i),
          op_name, ": key dim ", i, " (size ", key.size(i),
          ") is not broadcastable with output dim ", i,
          " (size ", self.size(i), ")");
    }

    key_dims = ndim;
    for (int64_t i = ndim - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      elems_per_key *= self.size(i);
      key_dims--;
    }
  } else {
    elems_per_key = self.numel();
  }

  int64_t num_keys = self.numel() / elems_per_key;
  auto output = self.contiguous();

  // When num_keys == 1, the kernel reads keys[0] and keys[1] directly,
  // so the key must be contiguous.  For multi-key, the OffsetCalculator
  // handles strided access.
  Tensor key_contig;
  if (num_keys == 1) {
    key_contig = key.contiguous();
  }
  const uint64_t* key_ptr = num_keys == 1
      ? key_contig.data_ptr<uint64_t>()
      : key.data_ptr<uint64_t>();

  // OffsetCalculator maps a linear key index to the element offset in the
  // key tensor.  Uses output sizes for index decomposition and key strides
  // for offset computation; broadcast dims (key size 1) get stride 0 so
  // all positions map to the same key.
  std::vector<int64_t> oc_sizes(key_dims);
  std::vector<int64_t> oc_strides(key_dims);
  for (int64_t i = 0; i < key_dims; i++) {
    int64_t dim = key_dims - 1 - i;
    oc_sizes[i] = self.size(dim);
    oc_strides[i] = key.size(dim) > 1 ? key.stride(dim) : 0;
  }
  const int64_t* oc_strides_ptr = oc_strides.data();
  auto key_offset_calc = OffsetCalculator<1>(
      key_dims, oc_sizes.data(), &oc_strides_ptr);

  constexpr int block_size = 256;
  int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  int max_blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm;

  constexpr int64_t elems_per_thread = 16;
  int num_blocks;
  if (num_keys == 1) {
    // Single-key: launch up to max occupancy, each thread divides work.
    num_blocks = std::min(
        static_cast<int>((elems_per_key + block_size - 1) / block_size),
        max_blocks);
  } else {
    int64_t num_chunks = (elems_per_key + elems_per_thread - 1) / elems_per_thread;
    int64_t total_threads = num_keys * num_chunks;
    num_blocks = std::min(
        static_cast<int>((total_threads + block_size - 1) / block_size),
        max_blocks);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), op_name, [&] {
    DistPolicyT<scalar_t> dist(param1, param2);

    // Shared memory for the single-key tiled path (non-double only).
    // Layout: warps_per_block * 32 threads * (EPT + 1 padding) floats.
    size_t smem_bytes = 0;
    if (num_keys == 1 && sizeof(scalar_t) <= sizeof(float)) {
      constexpr int PADDED = 4 * 8 + 1;  // elems_per_call * K + 1 = 33
      smem_bytes = (block_size / 32) * 32 * PADDED * sizeof(float);
    }

    if (num_keys == 1) {
      philox_distribution_kernel<scalar_t, true><<<num_blocks, block_size, smem_bytes,
          at::cuda::getCurrentCUDAStream()>>>(
          output.mutable_data_ptr<scalar_t>(),
          key_ptr, num_keys, elems_per_key, elems_per_thread, dist,
          key_offset_calc);
    } else {
      philox_distribution_kernel<scalar_t, false><<<num_blocks, block_size, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          output.mutable_data_ptr<scalar_t>(),
          key_ptr, num_keys, elems_per_key, elems_per_thread, dist,
          key_offset_calc);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }

  return self;
}

} // anonymous namespace

Tensor& _philox_uniform_cuda_(Tensor& self, const Tensor& key, double low, double high, bool portable) {
  return philox_distribution_impl<UniformPolicy>(self, key, low, high, portable);
}

Tensor& _philox_normal_cuda_(Tensor& self, const Tensor& key, double mean, double stddev, bool portable) {
  return philox_distribution_impl<NormalPolicy>(self, key, mean, stddev, portable);
}

} // namespace at::native
