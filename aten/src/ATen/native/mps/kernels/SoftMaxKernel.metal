#include <ATen/native/mps/kernels/SoftMaxKernel.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

static inline uint offset_a(uint row_idx, constant SoftmaxParams& p) {
  uint offset = 0;
  uint idx = row_idx;
  for (int d = int(p.ndim) - 2; d >= 0; d--) {
    uint coord = idx % p.outer_sizes[d];
    idx /= p.outer_sizes[d];
    offset += coord * p.outer_strides_a[d];
  }
  return offset;
}

static inline uint offset_b(uint row_idx, constant SoftmaxParams& p) {
  uint offset = 0;
  uint idx = row_idx;
  for (int d = int(p.ndim) - 2; d >= 0; d--) {
    uint coord = idx % p.outer_sizes[d];
    idx /= p.outer_sizes[d];
    offset += coord * p.outer_strides_b[d];
  }
  return offset;
}

static inline float4 load_vec4(device const float* p) {
  return *reinterpret_cast<device const packed_float4*>(p);
}
static inline float4 load_vec4(device const half* p) {
  return float4(*reinterpret_cast<device const packed_half4*>(p));
}
static inline float4 load_vec4(device const bfloat* p) {
  return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}

static inline void store_vec4(device float* p, float4 v) {
  *reinterpret_cast<device packed_float4*>(p) = v;
}
static inline void store_vec4(device half* p, float4 v) {
  *reinterpret_cast<device packed_half4*>(p) = half4(v);
}
static inline void store_vec4(device bfloat* p, float4 v) {
  p[0] = static_cast<bfloat>(v[0]);
  p[1] = static_cast<bfloat>(v[1]);
  p[2] = static_cast<bfloat>(v[2]);
  p[3] = static_cast<bfloat>(v[3]);
}

// Forward single-row: values cached in registers (1 read, 1 write).
// Reads from input using stride_a, writes to output contiguously.

template <typename T>
kernel void softmax_forward_single_row(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  device const T* x = input + offset_a(tg_id, params);
  device T* out = output + tg_id * axis_size;
  uint base = tid * N_READS;

  bool contiguous = (sa == 1);
  float vals[N_READS];
  float local_max = -INFINITY;
  if (base + N_READS <= axis_size) {
    if (contiguous) {
      float4 v = load_vec4(x + base);
      vals[0] = v.x;
      vals[1] = v.y;
      vals[2] = v.z;
      vals[3] = v.w;
    } else {
      for (int i = 0; i < N_READS; i++)
        vals[i] = float(x[(base + i) * sa]);
    }
    local_max = fmax(fmax(vals[0], vals[1]), fmax(vals[2], vals[3]));
  } else {
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (base + i < axis_size)
          ? (contiguous ? float(x[base + i]) : float(x[(base + i) * sa]))
          : -INFINITY;
      local_max = fmax(local_max, vals[i]);
    }
  }

  threadgroup float shared[simdgroup_size + 1];

  float sg_max = simd_max(local_max);
  if (simdgroup_id == 0)
    shared[simd_lane_id] = -INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0)
    shared[simdgroup_id] = sg_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    float row_max_candidate = shared[simd_lane_id];
    float global_max = simd_max(row_max_candidate);
    if (simd_lane_id == 0)
      shared[simdgroup_size] = global_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = shared[simdgroup_size];

  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    vals[i] = metal::precise::exp(vals[i] - row_max);
    local_sum += vals[i];
  }

  float sg_sum = simd_sum(local_sum);
  if (simdgroup_id == 0)
    shared[simd_lane_id] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0)
    shared[simdgroup_id] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    float global_sum = simd_sum(shared[simd_lane_id]);
    if (simd_lane_id == 0)
      shared[simdgroup_size] = global_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv_sum = 1.0f / shared[simdgroup_size];

  // out is always contiguous (output + tg_id * axis_size), so always use
  // store_vec4.
  if (base + N_READS <= axis_size) {
    store_vec4(
        out + base, float4(vals[0], vals[1], vals[2], vals[3]) * inv_sum);
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        out[base + i] = static_cast<T>(vals[i] * inv_sum);
    }
  }
}

// Forward looped: online softmax fuses max+sum into one pass over memory.

template <typename T>
kernel void softmax_forward_looped(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  device const T* x = input + offset_a(tg_id, params);
  device T* out = output + tg_id * axis_size;
  bool contiguous = (sa == 1);

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 v;
      if (contiguous) {
        v = load_vec4(x + base);
      } else {
        v = float4(
            x[base * sa],
            x[(base + 1) * sa],
            x[(base + 2) * sa],
            x[(base + 3) * sa]);
      }
      float chunk_max = fmax(fmax(v.x, v.y), fmax(v.z, v.w));
      float new_max = fmax(local_max, chunk_max);
      local_sum = local_sum * metal::precise::exp(local_max - new_max) +
          metal::precise::exp(v.x - new_max) +
          metal::precise::exp(v.y - new_max) +
          metal::precise::exp(v.z - new_max) +
          metal::precise::exp(v.w - new_max);
      local_max = new_max;
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        float new_max = fmax(local_max, val);
        local_sum = local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(val - new_max);
        local_max = new_max;
      }
    }
  }

  float sg_max = simd_max(local_max);
  local_sum *= metal::precise::exp(local_max - sg_max);
  float sg_sum = simd_sum(local_sum);

  threadgroup float shared_max[simdgroup_size];
  threadgroup float shared_sum[simdgroup_size];
  threadgroup float tg_result[2];

  if (simd_lane_id == 0) {
    shared_max[simdgroup_id] = sg_max;
    shared_sum[simdgroup_id] = sg_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    float m = shared_max[simd_lane_id];
    float global_max = simd_max(m);
    float s = shared_sum[simd_lane_id] * metal::precise::exp(m - global_max);
    float global_sum = simd_sum(s);
    if (simd_lane_id == 0) {
      tg_result[0] = global_max;
      tg_result[1] = global_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float row_max = tg_result[0];
  float inv_sum = 1.0f / tg_result[1];

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      if (contiguous) {
        store_vec4(
            out + base,
            metal::precise::exp(load_vec4(x + base) - row_max) * inv_sum);
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          out[base + i] = static_cast<T>(
              metal::precise::exp(float(x[(base + i) * sa]) - row_max) *
              inv_sum);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        out[i] = static_cast<T>(metal::precise::exp(val - row_max) * inv_sum);
      }
    }
  }
}

// Backward: grad_input = output * (grad_output - sum(grad_output * output))
// stride_a = grad_output strides, stride_b = output strides
// Writes grad_input contiguously.

template <typename T>
kernel void softmax_backward_single_row(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  device const T* dy = grad_output + offset_a(tg_id, params);
  device const T* y = output + offset_b(tg_id, params);
  device T* dx = grad_input + tg_id * axis_size;
  uint base = tid * N_READS;

  bool contiguous = (sa == 1) && (sb == 1);
  float dy_vals[N_READS];
  float y_vals[N_READS];
  float local_dot = 0.0f;
  if (base + N_READS <= axis_size) {
    if (contiguous) {
      float4 dy_v = load_vec4(dy + base);
      float4 y_v = load_vec4(y + base);
      dy_vals[0] = dy_v.x;
      dy_vals[1] = dy_v.y;
      dy_vals[2] = dy_v.z;
      dy_vals[3] = dy_v.w;
      y_vals[0] = y_v.x;
      y_vals[1] = y_v.y;
      y_vals[2] = y_v.z;
      y_vals[3] = y_v.w;
      local_dot = dot(dy_v, y_v);
    } else {
      for (int i = 0; i < N_READS; i++) {
        dy_vals[i] = float(dy[(base + i) * sa]);
        y_vals[i] = float(y[(base + i) * sb]);
        local_dot += dy_vals[i] * y_vals[i];
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size) {
        dy_vals[i] =
            contiguous ? float(dy[base + i]) : float(dy[(base + i) * sa]);
        y_vals[i] = contiguous ? float(y[base + i]) : float(y[(base + i) * sb]);
        local_dot += dy_vals[i] * y_vals[i];
      }
    }
  }

  threadgroup float shared_dot[simdgroup_size + 1];

  if (simdgroup_id == 0)
    shared_dot[simd_lane_id] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sg_dot = simd_sum(local_dot);
  if (simd_lane_id == 0)
    shared_dot[simdgroup_id] = sg_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    float d = simd_sum(shared_dot[simd_lane_id]);
    if (simd_lane_id == 0)
      shared_dot[simdgroup_size] = d;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot_sum = shared_dot[simdgroup_size];

  // dx is always contiguous (grad_input + tg_id * axis_size), so always use
  // store_vec4.
  if (base + N_READS <= axis_size) {
    store_vec4(
        dx + base,
        float4(y_vals[0], y_vals[1], y_vals[2], y_vals[3]) *
            (float4(dy_vals[0], dy_vals[1], dy_vals[2], dy_vals[3]) - dot_sum));
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        dx[base + i] = static_cast<T>(y_vals[i] * (dy_vals[i] - dot_sum));
    }
  }
}

// Backward looped: vectorized dot product with strided or contiguous access.

template <typename T>
kernel void softmax_backward_looped(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  device const T* dy = grad_output + offset_a(tg_id, params);
  device const T* y = output + offset_b(tg_id, params);
  device T* dx = grad_input + tg_id * axis_size;
  bool contiguous = (sa == 1) && (sb == 1);

  float local_dot = 0.0f;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      if (contiguous) {
        local_dot += dot(load_vec4(dy + base), load_vec4(y + base));
      } else {
        float4 dy_v = float4(
            dy[base * sa],
            dy[(base + 1) * sa],
            dy[(base + 2) * sa],
            dy[(base + 3) * sa]);
        float4 y_v = float4(
            y[base * sb],
            y[(base + 1) * sb],
            y[(base + 2) * sb],
            y[(base + 3) * sb]);
        local_dot += dot(dy_v, y_v);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++)
        local_dot += (contiguous ? float(dy[i]) : float(dy[i * sa])) *
            (contiguous ? float(y[i]) : float(y[i * sb]));
    }
  }

  local_dot = simd_sum(local_dot);

  threadgroup float shared_dot[simdgroup_size + 1];

  if (simd_lane_id == 0)
    shared_dot[simdgroup_id] = local_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    float d = simd_sum(shared_dot[simd_lane_id]);
    if (simd_lane_id == 0)
      shared_dot[simdgroup_size] = d;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot_sum = shared_dot[simdgroup_size];

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      if (contiguous) {
        float4 y_v = load_vec4(y + base);
        store_vec4(dx + base, y_v * (load_vec4(dy + base) - dot_sum));
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          dx[base + i] = static_cast<T>(
              float(y[(base + i) * sb]) *
              (float(dy[(base + i) * sa]) - dot_sum));
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float yi = contiguous ? float(y[i]) : float(y[i * sb]);
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        dx[i] = static_cast<T>(yi * (dyi - dot_sum));
      }
    }
  }
}

// Two-pass backward for low-occupancy cases (few rows, large axis).
// Phase 1: each threadgroup computes a partial dot(dy, y) over its chunk.
// Phase 2: each threadgroup sums partial dots, then computes grad_input for its
// chunk.

template <typename T>
kernel void softmax_backward_2pass_dot(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device float* partial_sums [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint num_chunks = params.num_chunks;
  uint chunk_id = tg_id % num_chunks;
  uint row_id = tg_id / num_chunks;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  device const T* dy = grad_output + offset_a(row_id, params);
  device const T* y = output + offset_b(row_id, params);
  bool contiguous = (sa == 1) && (sb == 1);

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  float local_dot = 0.0f;
  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
      if (contiguous) {
        local_dot += dot(load_vec4(dy + base), load_vec4(y + base));
      } else {
        float4 dy_v = float4(
            dy[base * sa],
            dy[(base + 1) * sa],
            dy[(base + 2) * sa],
            dy[(base + 3) * sa]);
        float4 y_v = float4(
            y[base * sb],
            y[(base + 1) * sb],
            y[(base + 2) * sb],
            y[(base + 3) * sb]);
        local_dot += dot(dy_v, y_v);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++)
        local_dot += (contiguous ? float(dy[i]) : float(dy[i * sa])) *
            (contiguous ? float(y[i]) : float(y[i * sb]));
    }
  }

  local_dot = simd_sum(local_dot);

  threadgroup float shared_dot[simdgroup_size];

  if (simd_lane_id == 0)
    shared_dot[simdgroup_id] = local_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    float d = simd_sum(shared_dot[simd_lane_id]);
    if (simd_lane_id == 0)
      partial_sums[row_id * num_chunks + chunk_id] = d;
  }
}

template <typename T>
kernel void softmax_backward_2pass_grad(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    device const float* partial_sums [[buffer(3)]],
    constant SoftmaxParams& params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;
  uint num_chunks = params.num_chunks;
  uint chunk_id = tg_id % num_chunks;
  uint row_id = tg_id / num_chunks;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  device const T* dy = grad_output + offset_a(row_id, params);
  device const T* y = output + offset_b(row_id, params);
  device T* dx = grad_input + row_id * axis_size;
  bool contiguous = (sa == 1) && (sb == 1);

  float dot_sum = 0.0f;
  for (uint i = 0; i < num_chunks; i++)
    dot_sum += partial_sums[row_id * num_chunks + i];

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
      if (contiguous) {
        float4 y_v = load_vec4(y + base);
        store_vec4(dx + base, y_v * (load_vec4(dy + base) - dot_sum));
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          dx[base + i] = static_cast<T>(
              float(y[(base + i) * sb]) *
              (float(dy[(base + i) * sa]) - dot_sum));
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float yi = contiguous ? float(y[i]) : float(y[i * sb]);
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        dx[i] = static_cast<T>(yi * (dyi - dot_sum));
      }
    }
  }
}

// Template instantiations

#define instantiate_softmax_forward_single_row(DTYPE)                     \
  template [[host_name("softmax_forward_single_row_" #DTYPE)]] [[kernel]] \
  void softmax_forward_single_row<DTYPE>(                                 \
      device const DTYPE* input [[buffer(0)]],                            \
      device DTYPE* output [[buffer(1)]],                                 \
      constant SoftmaxParams& params [[buffer(2)]],                       \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint simd_lane_id [[thread_index_in_simdgroup]],                    \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_forward_looped(DTYPE)                          \
  template [[host_name("softmax_forward_looped_" #DTYPE)]] [[kernel]] void \
  softmax_forward_looped<DTYPE>(                                           \
      device const DTYPE* input [[buffer(0)]],                             \
      device DTYPE* output [[buffer(1)]],                                  \
      constant SoftmaxParams& params [[buffer(2)]],                        \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint lsize [[threads_per_threadgroup]],                              \
      uint simd_lane_id [[thread_index_in_simdgroup]],                     \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_backward_single_row(DTYPE)                     \
  template [[host_name("softmax_backward_single_row_" #DTYPE)]] [[kernel]] \
  void softmax_backward_single_row<DTYPE>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                       \
      device const DTYPE* output [[buffer(1)]],                            \
      device DTYPE* grad_input [[buffer(2)]],                              \
      constant SoftmaxParams& params [[buffer(3)]],                        \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint simd_lane_id [[thread_index_in_simdgroup]],                     \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_backward_looped(DTYPE)                          \
  template [[host_name("softmax_backward_looped_" #DTYPE)]] [[kernel]] void \
  softmax_backward_looped<DTYPE>(                                           \
      device const DTYPE* grad_output [[buffer(0)]],                        \
      device const DTYPE* output [[buffer(1)]],                             \
      device DTYPE* grad_input [[buffer(2)]],                               \
      constant SoftmaxParams& params [[buffer(3)]],                         \
      uint tg_id [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint lsize [[threads_per_threadgroup]],                               \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_backward_2pass_dot(DTYPE)                          \
  template [[host_name("softmax_backward_2pass_dot_" #DTYPE)]] [[kernel]] void \
  softmax_backward_2pass_dot<DTYPE>(                                           \
      device const DTYPE* grad_output [[buffer(0)]],                           \
      device const DTYPE* output [[buffer(1)]],                                \
      device float* partial_sums [[buffer(2)]],                                \
      constant SoftmaxParams& params [[buffer(3)]],                            \
      uint tg_id [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint lsize [[threads_per_threadgroup]],                                  \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_backward_2pass_grad(DTYPE)                     \
  template                                                                 \
      [[host_name("softmax_backward_2pass_grad_" #DTYPE)]] [[kernel]] void \
      softmax_backward_2pass_grad<DTYPE>(                                  \
          device const DTYPE* grad_output [[buffer(0)]],                   \
          device const DTYPE* output [[buffer(1)]],                        \
          device DTYPE* grad_input [[buffer(2)]],                          \
          device const float* partial_sums [[buffer(3)]],                  \
          constant SoftmaxParams& params [[buffer(4)]],                    \
          uint tg_id [[threadgroup_position_in_grid]],                     \
          uint tid [[thread_position_in_threadgroup]],                     \
          uint lsize [[threads_per_threadgroup]]);

#define instantiate_softmax(DTYPE)                              \
  instantiate_softmax_forward_single_row(DTYPE)                 \
      instantiate_softmax_forward_looped(DTYPE)                 \
          instantiate_softmax_backward_single_row(DTYPE)        \
              instantiate_softmax_backward_looped(DTYPE)        \
                  instantiate_softmax_backward_2pass_dot(DTYPE) \
                      instantiate_softmax_backward_2pass_grad(DTYPE)

instantiate_softmax(float);
instantiate_softmax(half);
instantiate_softmax(bfloat);
