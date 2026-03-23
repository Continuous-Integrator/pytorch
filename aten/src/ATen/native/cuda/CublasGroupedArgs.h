#pragma once

#include <cuda.h>

#include <ATen/BlasBackend.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

namespace at::native {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
struct cublasGroupedArgs {
  cublasGroupedArgs(
      const Tensor& mat1,
      const Tensor& mat2,
      const std::optional<Tensor>& offs,
      Tensor& c,
      const std::optional<Tensor>& scale_a = std::nullopt,
      const std::optional<Tensor>& scale_b = std::nullopt,
      const std::optional<Tensor>& scale_result = std::nullopt,
      const std::optional<at::blas::ScalingType>& scaling_choice_a = std::nullopt,
      const std::optional<at::blas::ScalingType>& scaling_choice_b = std::nullopt);

  char transa, transb;
  int64_t avgM, avgN, avgK;
  ScalarType A_dtype, B_dtype, result_dtype;
  int batchCount;

  // All arrays live in a single device allocation
  Tensor buf;
  // BlockWise1x32 scale pointer arrays need a separate allocation
  Tensor scale_ptr_buf;
  void* mArray;
  void* nArray;
  void* kArray;
  void* ldaArray;
  void* ldbArray;
  void* lddArray;
  void* APtrArray;
  void* BPtrArray;
  void* DPtrArray;
  void* alphaPtrArray;
  void* betaPtrArray;

  void* scale_mata_ptr = nullptr;
  void* scale_matb_ptr = nullptr;
  void* scale_result_ptr = nullptr;
  at::blas::ScalingType scale_mata_scaling_type = at::blas::ScalingType::TensorWise;
  at::blas::ScalingType scale_matb_scaling_type = at::blas::ScalingType::TensorWise;
  c10::ScalarType scale_mata_dtype = c10::ScalarType::Float;
  c10::ScalarType scale_matb_dtype = c10::ScalarType::Float;
};
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020

} // namespace at::native
