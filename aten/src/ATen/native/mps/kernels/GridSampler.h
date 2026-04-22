#pragma once
#include <c10/metal/common.h>

#ifdef __METAL_VERSION__
// Must match at::native::detail::GridSamplerInterpolation
enum class GridSamplerInterpolation : int32_t {
  Bilinear = 0,
  Nearest = 1,
  Bicubic = 2,
};

// Must match at::native::detail::GridSamplerPadding
enum class GridSamplerPadding : int32_t {
  Zeros = 0,
  Border = 1,
  Reflection = 2,
};
#else
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/Pool.h>
using at::native::GridSamplerInterpolation;
using at::native::GridSamplerPadding;
#endif

template <unsigned N = 5, typename idx_type_t = int32_t>
struct GridSamplerParams {
  int32_t sampler_dims;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<idx_type_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<idx_type_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> grid_sizes;
  ::c10::metal::array<idx_type_t, N> grid_strides;
  bool align_corners;

#ifndef __METAL_VERSION__
  GridSamplerParams() = default;
  GridSamplerParams(
      const at::TensorBase& output,
      const at::TensorBase& input,
      const at::TensorBase& grid,
      bool align_corners_)
      : sampler_dims(N - 2), align_corners(align_corners_) {
    using at::native::safe_downcast;
    for (unsigned dim = 0; dim < N; dim++) {
      output_sizes[dim] = safe_downcast<idx_type_t>(output.size(dim));
      output_strides[dim] = safe_downcast<idx_type_t>(output.stride(dim));
      input_sizes[dim] = safe_downcast<idx_type_t>(input.size(dim));
      input_strides[dim] = safe_downcast<idx_type_t>(input.stride(dim));
      grid_sizes[dim] = safe_downcast<idx_type_t>(grid.size(dim));
      grid_strides[dim] = safe_downcast<idx_type_t>(grid.stride(dim));
    }
  }
#endif
};

template <unsigned N = 5, typename idx_type_t = int32_t>
struct GridSamplerBackwardParams {
  GridSamplerParams<N, idx_type_t> forward;
  ::c10::metal::array<idx_type_t, N> grad_output_strides;
  ::c10::metal::array<idx_type_t, N> grad_input_strides;
  idx_type_t grad_grid_sW;
  GridSamplerPadding padding_mode;
};

struct GridSampler3DBackwardParams {
  GridSamplerInterpolation interpolation_mode;
  GridSamplerPadding padding_mode;
  bool align_corners;
  bool compute_grad_input;
  bool compute_grad_grid;
  ::c10::metal::array<int32_t, 5> input_sizes;
  ::c10::metal::array<int32_t, 5> output_sizes;
  ::c10::metal::array<int32_t, 5> input_strides;
  ::c10::metal::array<int32_t, 5> grad_input_strides;
  ::c10::metal::array<int32_t, 5> grad_grid_strides;
  ::c10::metal::array<int32_t, 5> grid_strides;
  ::c10::metal::array<int32_t, 5> grad_output_strides;
};
