#pragma once

#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/CublasGroupedArgs.cuh>

namespace at::native {

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace {

// TODO: https://github.com/pytorch/pytorch/pull/59380#pullrequestreview-725310492
c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(const Tensor& tensor, bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, true);
  }

  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

} // namespace

/**
 * @brief Prepares matrices for CUBLAS operation
 *
 * This constructor prepares tensors for CUBLAS
 * The main difference is that PyTorch uses row-major as the default and
 * CUBLAS expects column-major.
 *
 * @details
 * To enable row-major output while using CUBLAS,
 * we use the mathematical identity that (A × B)^T = B^T × A^T.
 *
 * Transpose in this context refers to Cublas's(Fortran) definition of transpose (row-major)
 * T = row-major, N = col-major
 *
 * Example:
 * For matrices A (M×K)(row-major) and B (K×N)(row-major):
 *   - Standard multiplication: A × B = (M×K) × (K×N) = M×N result (row-major)
 *   - Using our transpose trick: (B^T × A^T) = (N×K)(T) × (K×M)(T) = N×M(N)
 *   - However, since the output form cublas is column-major this is
 *   - equivalent to an output of size MxN row-major as expected
 *
 * The transpose flags are derived from the layouts of the passed in tensors
 *
 * If the operands are in packed float4 format, `k`, `lda` and `ldb` are adjusted
 * to their unpacked values to match what cuBLAS expects.
 *
 * @param mat1 First input matrix
 * @param mat2 Second input matrix
 * @param c Output matrix (result)
 * @param scale_a Optional scaling factor for first matrix
 * @param scale_b Optional scaling factor for second matrix
 * @param scale_result Optional scaling factor for result
 */
struct cublasCommonArgs {
  cublasCommonArgs(
      const Tensor& mat1,
      const Tensor& mat2,
      Tensor& c,
      const std::optional<Tensor>& scale_a = std::nullopt,
      const std::optional<Tensor>& scale_b = std::nullopt,
      const std::optional<Tensor>& scale_result = std::nullopt,
      const std::optional<ScalingType>& scaling_choice_a = std::nullopt,
      const std::optional<ScalingType>& scaling_choice_b = std::nullopt) {
    bool transpose_result = false, transpose_a = false, transpose_b = false;
    result = prepare_matrix_for_cublas(c, transpose_result);
    mata = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_a, transpose_result);
    matb = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_b, transpose_result);

    // Handle scale tensors if provided
    if (scale_a && scale_b) {
      // By default since we return in row-major we run the gemm
      // as B.T @ A.T, check transpose_result to determine if we flip the scales
      scale_mata_ptr = transpose_result ? scale_b->data_ptr() : scale_a->data_ptr();
      scale_mata_dtype = transpose_result ? scale_b->scalar_type() : scale_a->scalar_type();
      scaling_mata_type = transpose_result ? scaling_choice_b : scaling_choice_a;
      scale_matb_ptr = transpose_result ? scale_a->data_ptr() : scale_b->data_ptr();
      scale_matb_dtype = transpose_result ? scale_a->scalar_type() : scale_b->scalar_type();
      scaling_matb_type = transpose_result ? scaling_choice_a : scaling_choice_b;
    }

    if (scale_result) {
      scale_result_ptr = scale_result->data_ptr();
      scale_result_dtype = scale_result->scalar_type();
    }

    // Update transpose flags
    if (transpose_result) {
      transpose_a = !transpose_a;
      transpose_b = !transpose_b;
    }

    auto sizes_a = mata->sizes();
    auto sizes_b = matb->sizes();

    m = sizes_a[transpose_result ? 1 : 0];
    k = sizes_a[transpose_result ? 0 : 1];
    n = sizes_b[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_a == transpose_result) ? 1 : 0);
    ldb = matb->stride((transpose_b == transpose_result) ? 1 : 0);
    result_ld = result->stride(transpose_result ? 0 : 1);
    transa = transpose_a ? mata->is_conj() ? 'c' : 't' : 'n';
    transb = transpose_b ? matb->is_conj() ? 'c' : 't' : 'n';

    // cuBLAS expects unpacked values of `k`, `lda` and `ldb`, adjust for 4x2 packing
    // if the gemm operands are in packed float4
    if (mat1.dtype() == at::kFloat4_e2m1fn_x2 && mat2.dtype() == at::kFloat4_e2m1fn_x2) {
      k = k * 2;
      lda = lda * 2;
      ldb = ldb * 2;
    }
  }

  // Matrix members
  char transa, transb;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<Tensor> mata, matb, result;

  // Scale members
  void* scale_mata_ptr = nullptr;
  void* scale_matb_ptr = nullptr;
  void* scale_result_ptr = nullptr;
  std::optional<c10::ScalarType> scale_mata_dtype;
  std::optional<ScalingType> scaling_mata_type;
  std::optional<c10::ScalarType> scale_matb_dtype;
  std::optional<ScalingType> scaling_matb_type;
  std::optional<c10::ScalarType> scale_result_dtype;
};

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
struct cublasCommonGroupedArgs {
  cublasCommonGroupedArgs(
      const Tensor& mat1,
      const Tensor& mat2,
      const std::optional<Tensor>& offs,
      Tensor& c,
      const std::optional<Tensor>& scale_a = std::nullopt,
      const std::optional<Tensor>& scale_b = std::nullopt,
      const std::optional<Tensor>& scale_result = std::nullopt,
      const std::optional<ScalingType>& scaling_choice_a = std::nullopt,
      const std::optional<ScalingType>& scaling_choice_b = std::nullopt) {
        const bool a_is_2d = mat1.dim() == 2;
        const bool b_is_2d = mat2.dim() == 2;
        if (a_is_2d || b_is_2d) {
          TORCH_CHECK(offs.has_value(), "Offsets tensor must be provided when at least one input is 2D");
        }

        A_dtype = mat2.scalar_type();
        B_dtype = mat1.scalar_type();
        result_dtype = c.scalar_type();
        const int64_t esz = mat1.element_size();
        const int64_t out_esz = c.element_size();

        if (offs.has_value()) {
          batchCount = offs.value().size(0);
        } else {
          batchCount = mat1.size(0);
        }

        // cuBLAS is column-major. To get a row-major result C = mat1 × mat2,
        // we use the identity C^T = mat2^T × mat1^T. So cuBLAS-A = mat2 and
        // cuBLAS-B = mat1. The transpose flags depend on inner-dim layout:
        //   row-major (stride(-1)==1): cuBLAS sees it as col-major "already
        //     transposed" → after the B^T×A^T flip, the op flag is 'n'
        //   col-major (stride(-2)==1): cuBLAS sees it naturally → after
        //     the flip, the op flag is 't'
        const bool mat2_row_major = mat2.stride(-1) == 1;
        const bool mat1_row_major = mat1.stride(-1) == 1;
        transa = mat2_row_major ? 'n' : 't';
        transb = mat1_row_major ? 'n' : 't';

        // User-space dimensions
        const int64_t user_M = mat1.size(-2);
        const int64_t user_N = mat2.size(-1);
        const int64_t user_K = mat1.size(-1);

        // In the cuBLAS B^T×A^T convention:
        //   cublas_m = user_N, cublas_n = user_M, cublas_k = user_K
        const int32_t cublas_m = static_cast<int32_t>(user_N);
        const int32_t cublas_n = static_cast<int32_t>(user_M);
        const int32_t cublas_k = static_cast<int32_t>(user_K);

        // Leading dimensions (constant across groups, from inner-dim strides)
        // cuBLAS-A = mat2, cuBLAS-B = mat1
        const int32_t lda_val = static_cast<int32_t>(transa == 't' ? mat2.stride(-1) : mat2.stride(-2));
        const int32_t ldb_val = static_cast<int32_t>(transb == 't' ? mat1.stride(-1) : mat1.stride(-2));
        const int32_t ldd_val = static_cast<int32_t>(c.stride(-2));

        if (scale_a && scale_b) {
          scale_mata_ptr = scale_b->data_ptr();
          scale_matb_ptr = scale_a->data_ptr();
          scale_mata_dtype = scale_b->scalar_type();
          scale_matb_dtype = scale_a->scalar_type();

          auto infer = [&](const Tensor& scale) -> at::blas::ScalingType {
            if (scale.scalar_type() == at::kFloat8_e8m0fnu)
              return at::blas::ScalingType::BlockWise1x32;
            if (scale.numel() == 1)
              return at::blas::ScalingType::TensorWise;
            return at::blas::ScalingType::GroupWise;
          };
          // mata corresponds to scale_b (cuBLAS-A = mat2)
          scale_mata_scaling_type = scaling_choice_a.value_or(infer(*scale_b));
          // matb corresponds to scale_a (cuBLAS-B = mat1)
          scale_matb_scaling_type = scaling_choice_b.value_or(infer(*scale_a));
        }
        if (scale_result) {
          scale_result_ptr = scale_result->data_ptr();
        }

        // GroupWise scales need device-side pointer arrays (one pointer per
        // group) because cuBLAS PER_BATCH_SCALAR mode expects the scale
        // pointer to be an array of device pointers.
        const bool mata_groupwise = scale_mata_scaling_type == at::blas::ScalingType::GroupWise;
        const bool matb_groupwise = scale_matb_scaling_type == at::blas::ScalingType::GroupWise;

        // Determine per-case which dimensions are variable (delta-based)
        // and how pointer strides work
        bool m_is_delta = false, n_is_delta = false, k_is_delta = false;
        int64_t a_offs_stride = 0, a_idx_stride = 0;
        int64_t b_offs_stride = 0, b_idx_stride = 0;
        int64_t d_offs_stride = 0, d_idx_stride = 0;

        if (a_is_2d && b_is_2d) {
          // 2D x 2D: jagged K
          k_is_delta = true;
          a_offs_stride = mat2.stride(-2) * esz;
          b_offs_stride = mat1.stride(-1) * esz;
          d_idx_stride = c.stride(0) * out_esz;
          avgM = cublas_m;
          avgN = cublas_n;
          avgK = user_K / batchCount;
        } else if (a_is_2d && !b_is_2d) {
          // 2D x 3D: jagged M (user M varies, cublas n varies)
          n_is_delta = true;
          a_idx_stride = mat2.stride(0) * esz;
          b_offs_stride = mat1.stride(-2) * esz;
          d_offs_stride = c.stride(-2) * out_esz;
          avgM = cublas_m;
          avgN = user_M / batchCount;
          avgK = cublas_k;
        } else if (!a_is_2d && b_is_2d) {
          // 3D x 2D: jagged N (user N varies, cublas m varies)
          m_is_delta = true;
          a_offs_stride = mat2.stride(-1) * esz;
          b_idx_stride = mat1.stride(0) * esz;
          d_offs_stride = c.stride(-1) * out_esz;
          avgM = user_N / batchCount;
          avgN = cublas_n;
          avgK = cublas_k;
        } else {
          // 3D x 3D: all dimensions fixed
          a_idx_stride = mat2.stride(0) * esz;
          b_idx_stride = mat1.stride(0) * esz;
          d_idx_stride = c.stride(0) * out_esz;
          avgM = cublas_m;
          avgN = cublas_n;
          avgK = cublas_k;
        }

        // Single device allocation for all arrays:
        //   6 x int32[batchCount]  = batchCount * 24 bytes  (m,n,k,lda,ldb,ldd)
        //   5 x int64[batchCount]  = batchCount * 40 bytes  (A,B,D,alpha,beta ptrs)
        //   2 x float              = 8 bytes          (alpha, beta)
        // + optionally up to 2 x int64[batchCount] for per-group scale pointer arrays
        const int extra_ptr_arrays = (mata_groupwise ? 1 : 0) + (matb_groupwise ? 1 : 0);
        const int64_t buf_bytes = static_cast<int64_t>(batchCount) * 64 + 8
            + static_cast<int64_t>(extra_ptr_arrays) * batchCount * 8;
        buf = at::empty({buf_bytes}, mat1.options().dtype(at::kByte));
        char* base = static_cast<char*>(buf.data_ptr());

        // int32 arrays (6 x batchCount)
        mArray   = reinterpret_cast<void*>(base);
        nArray   = reinterpret_cast<void*>(base + batchCount * 4);
        kArray   = reinterpret_cast<void*>(base + batchCount * 8);
        ldaArray = reinterpret_cast<void*>(base + batchCount * 12);
        ldbArray = reinterpret_cast<void*>(base + batchCount * 16);
        lddArray = reinterpret_cast<void*>(base + batchCount * 20);

        // int64 arrays (5 x batchCount), starting at offset batchCount * 24
        APtrArray     = reinterpret_cast<void*>(base + batchCount * 24);
        BPtrArray     = reinterpret_cast<void*>(base + batchCount * 32);
        DPtrArray     = reinterpret_cast<void*>(base + batchCount * 40);
        alphaPtrArray = reinterpret_cast<void*>(base + batchCount * 48);
        betaPtrArray  = reinterpret_cast<void*>(base + batchCount * 56);

        // Alpha/beta scalars at the end of the fixed-size region
        float* alpha_scalar = reinterpret_cast<float*>(base + batchCount * 64);
        float* beta_scalar  = reinterpret_cast<float*>(base + batchCount * 64 + 4);

        // Optional per-group scale pointer arrays follow alpha/beta
        int64_t extra_offset = static_cast<int64_t>(batchCount) * 64 + 8;
        int64_t* scaleAPtrArray = nullptr;
        int64_t* scaleBPtrArray = nullptr;
        if (mata_groupwise) {
          scaleAPtrArray = reinterpret_cast<int64_t*>(base + extra_offset);
          extra_offset += batchCount * 8;
        }
        if (matb_groupwise) {
          scaleBPtrArray = reinterpret_cast<int64_t*>(base + extra_offset);
          extra_offset += batchCount * 8;
        }

        // Base addresses for scale data (cuBLAS-A = mat2 → scale_b, cuBLAS-B = mat1 → scale_a)
        const int64_t base_scale_a = scale_b ? reinterpret_cast<int64_t>(scale_b->data_ptr()) : 0;
        const int64_t base_scale_b = scale_a ? reinterpret_cast<int64_t>(scale_a->data_ptr()) : 0;

        const int64_t base_A = reinterpret_cast<int64_t>(mat2.data_ptr());
        const int64_t base_B = reinterpret_cast<int64_t>(mat1.data_ptr());
        const int64_t base_D = reinterpret_cast<int64_t>(c.data_ptr());

        const int32_t* offs_ptr = offs.has_value()
            ? static_cast<const int32_t*>(offs.value().data_ptr())
            : nullptr;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        launch_populate_cublas_grouped_args(
              batchCount, offs_ptr,
              base_A, base_B, base_D,
              cublas_m, cublas_n, cublas_k,
              m_is_delta, n_is_delta, k_is_delta,
              lda_val, ldb_val, ldd_val,
              a_offs_stride, a_idx_stride,
              b_offs_stride, b_idx_stride,
              d_offs_stride, d_idx_stride,
              static_cast<int32_t*>(mArray),
              static_cast<int32_t*>(nArray),
              static_cast<int32_t*>(kArray),
              static_cast<int32_t*>(ldaArray),
              static_cast<int32_t*>(ldbArray),
              static_cast<int32_t*>(lddArray),
              static_cast<int64_t*>(APtrArray),
              static_cast<int64_t*>(BPtrArray),
              static_cast<int64_t*>(DPtrArray),
              static_cast<int64_t*>(alphaPtrArray),
              static_cast<int64_t*>(betaPtrArray),
              alpha_scalar, beta_scalar,
              base_scale_a, base_scale_b,
              scaleAPtrArray, scaleBPtrArray,
              stream);

        // For GroupWise scales, point to the device-side pointer arrays
        // instead of the raw data pointer
        if (mata_groupwise) {
          scale_mata_ptr = scaleAPtrArray;
        }
        if (matb_groupwise) {
          scale_matb_ptr = scaleBPtrArray;
        }
  }

  char transa, transb;
  int64_t avgM, avgN, avgK;
  ScalarType A_dtype, B_dtype, result_dtype;
  int batchCount;

  // All arrays live in a single device allocation
  Tensor buf;
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
