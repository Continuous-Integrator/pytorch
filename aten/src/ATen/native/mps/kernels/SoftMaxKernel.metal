#include <c10/metal/common.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

// Forward single-row: values cached in registers (1 read, 1 write).

template <typename T>
kernel void softmax_forward_single_row(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint& axis_size [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint row_offset = tg_id * axis_size;
  device const T* x = input + row_offset;
  device T* out = output + row_offset;
  uint base = tid * N_READS;

  float vals[N_READS];
  float local_max = -INFINITY;
  if (base + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++)
      vals[i] = float(x[base + i]);
    local_max = fmax(fmax(vals[0], vals[1]), fmax(vals[2], vals[3]));
  } else {
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (base + i < axis_size) ? float(x[base + i]) : -INFINITY;
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
    vals[i] = metal::fast::exp(vals[i] - row_max);
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

  if (base + N_READS <= axis_size) {
#pragma unroll
    for (int i = 0; i < N_READS; i++)
      out[base + i] = static_cast<T>(vals[i] * inv_sum);
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
    constant uint& axis_size [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint row_offset = tg_id * axis_size;
  device const T* x = input + row_offset;
  device T* out = output + row_offset;

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 v = float4(x[base], x[base + 1], x[base + 2], x[base + 3]);
      float chunk_max = fmax(fmax(v.x, v.y), fmax(v.z, v.w));
      float new_max = fmax(local_max, chunk_max);
      local_sum = local_sum * metal::fast::exp(local_max - new_max) +
          metal::fast::exp(v.x - new_max) + metal::fast::exp(v.y - new_max) +
          metal::fast::exp(v.z - new_max) + metal::fast::exp(v.w - new_max);
      local_max = new_max;
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float val = float(x[i]);
        float new_max = fmax(local_max, val);
        local_sum = local_sum * metal::fast::exp(local_max - new_max) +
            metal::fast::exp(val - new_max);
        local_max = new_max;
      }
    }
  }

  float sg_max = simd_max(local_max);
  local_sum *= metal::fast::exp(local_max - sg_max);
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
    float s = shared_sum[simd_lane_id] * metal::fast::exp(m - global_max);
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
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        out[base + i] =
            static_cast<T>(metal::fast::exp(float(x[base + i]) - row_max) * inv_sum);
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++)
        out[i] = static_cast<T>(metal::fast::exp(float(x[i]) - row_max) * inv_sum);
    }
  }
}

// Backward: grad_input = output * (grad_output - sum(grad_output * output))
// Single-row: values cached in registers.

template <typename T>
kernel void softmax_backward_single_row(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint row_offset = tg_id * axis_size;
  device const T* dy = grad_output + row_offset;
  device const T* y = output + row_offset;
  device T* dx = grad_input + row_offset;
  uint base = tid * N_READS;

  float dy_vals[N_READS];
  float y_vals[N_READS];
  float local_dot = 0.0f;
  if (base + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      dy_vals[i] = float(dy[base + i]);
      y_vals[i] = float(y[base + i]);
      local_dot += dy_vals[i] * y_vals[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size) {
        dy_vals[i] = float(dy[base + i]);
        y_vals[i] = float(y[base + i]);
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

  if (base + N_READS <= axis_size) {
#pragma unroll
    for (int i = 0; i < N_READS; i++)
      dx[base + i] = static_cast<T>(y_vals[i] * (dy_vals[i] - dot_sum));
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        dx[base + i] = static_cast<T>(y_vals[i] * (dy_vals[i] - dot_sum));
    }
  }
}

// Backward looped: vectorized float4 dot product.

template <typename T>
kernel void softmax_backward_looped(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint row_offset = tg_id * axis_size;
  device const T* dy = grad_output + row_offset;
  device const T* y = output + row_offset;
  device T* dx = grad_input + row_offset;

  float local_dot = 0.0f;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 dy_v = float4(dy[base], dy[base + 1], dy[base + 2], dy[base + 3]);
      float4 y_v = float4(y[base], y[base + 1], y[base + 2], y[base + 3]);
      local_dot += dot(dy_v, y_v);
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++)
        local_dot += float(dy[i]) * float(y[i]);
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
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        dx[base + i] = static_cast<T>(
            float(y[base + i]) * (float(dy[base + i]) - dot_sum));
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++)
        dx[i] = static_cast<T>(float(y[i]) * (float(dy[i]) - dot_sum));
    }
  }
}

// Template instantiations

#define instantiate_softmax_forward_single_row(DTYPE)                     \
  template [[host_name("softmax_forward_single_row_" #DTYPE)]] [[kernel]] \
  void softmax_forward_single_row<DTYPE>(                                 \
      device const DTYPE* input [[buffer(0)]],                            \
      device DTYPE* output [[buffer(1)]],                                 \
      constant uint& axis_size [[buffer(2)]],                             \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint simd_lane_id [[thread_index_in_simdgroup]],                    \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_forward_looped(DTYPE)                          \
  template [[host_name("softmax_forward_looped_" #DTYPE)]] [[kernel]] void \
  softmax_forward_looped<DTYPE>(                                           \
      device const DTYPE* input [[buffer(0)]],                             \
      device DTYPE* output [[buffer(1)]],                                  \
      constant uint& axis_size [[buffer(2)]],                              \
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
      constant uint& axis_size [[buffer(3)]],                              \
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
      constant uint& axis_size [[buffer(3)]],                               \
      uint tg_id [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint lsize [[threads_per_threadgroup]],                               \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax(DTYPE)                       \
  instantiate_softmax_forward_single_row(DTYPE)          \
      instantiate_softmax_forward_looped(DTYPE)          \
          instantiate_softmax_backward_single_row(DTYPE) \
              instantiate_softmax_backward_looped(DTYPE)

instantiate_softmax(float);
instantiate_softmax(half);
instantiate_softmax(bfloat);
