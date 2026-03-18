#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

namespace c10d::nccl_extension {

TORCH_API bool is_nccl_symmem_available();

TORCH_API void nccl_put(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_get(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_wait_for_signal(at::Tensor& sigpad, int64_t signal);

TORCH_API void nccl_put_with_signal(
    at::Tensor& tensor,
    int64_t signal,
    int64_t peer);

// Simultaneously reduce N strided 2-D tensors, routing each to a specific
// destination rank. All tensors must be views of the same NCCL symmetric
// memory allocation with the same shape and outer stride.
TORCH_API void nccl_grouped_strided_reduce(
    at::TensorList inputs,
    at::IntArrayRef dst_ranks,
    const std::string& group_name,
    at::TensorList out = {});
} // namespace c10d::nccl_extension
