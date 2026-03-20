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

// Simultaneously reduce N column-block 2-D tensors from a shared input buffer,
// routing each to a specific destination rank. Column blocks are described by
// inclusive-prefix-sum offsets; all blocks must have equal width.
TORCH_API void nccl_reduce_scatter_columns(
    const at::Tensor& input,
    at::TensorList out,
    const std::string& group_name,
    std::optional<at::IntArrayRef> offsets,
    std::optional<at::IntArrayRef> dst_ranks,
    const std::string& red_op);
} // namespace c10d::nccl_extension
