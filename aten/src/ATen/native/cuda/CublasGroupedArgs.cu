#include <ATen/native/cuda/CublasGroupedArgs.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

namespace at::native {

// cuBLAS VEC32_UE8M0 scale tensor size (bytes, since e8m0 is 1 byte each).
// Mirrors getScaleTensorSize() from the cuBLAS samples.
__device__ __forceinline__ int64_t cublas_vec32_scale_size(int inner, int outer) {
  const int BLOCK_ROWS = 128; // S_BLOCK_INNER(4) * S_VSCALE(32)
  const int BLOCK_COLS = 128; // S_BLOCK_COLS(32) * S_BLOCK_ROWS(4)
  const int S_VSCALE = 32;
  int64_t s_rows = ((inner + BLOCK_ROWS - 1) / BLOCK_ROWS) * (BLOCK_ROWS / S_VSCALE);
  int64_t s_cols = ((outer + BLOCK_COLS - 1) / BLOCK_COLS) * BLOCK_COLS;
  return s_rows * s_cols;
}

__global__ void populate_cublas_grouped_args_kernel(
    const int32_t* __restrict__ offs,
    int64_t base_A, int64_t base_B, int64_t base_D,
    int32_t cublas_m, int32_t cublas_n, int32_t cublas_k,
    bool m_is_delta, bool n_is_delta, bool k_is_delta,
    int32_t lda_val, int32_t ldb_val, int32_t ldd_val,
    int64_t a_offs_stride, int64_t a_idx_stride,
    int64_t b_offs_stride, int64_t b_idx_stride,
    int64_t d_offs_stride, int64_t d_idx_stride,
    int32_t* __restrict__ m_out, int32_t* __restrict__ n_out, int32_t* __restrict__ k_out,
    int32_t* __restrict__ lda_out, int32_t* __restrict__ ldb_out, int32_t* __restrict__ ldd_out,
    int64_t* __restrict__ APtr_out, int64_t* __restrict__ BPtr_out, int64_t* __restrict__ DPtr_out,
    int64_t* __restrict__ alphaPtr_out, int64_t* __restrict__ betaPtr_out,
    float* __restrict__ alpha_ptr, float* __restrict__ beta_ptr,
    int64_t base_scale_a, int64_t base_scale_b,
    int64_t scale_a_stride_bytes, int64_t scale_b_stride_bytes,
    int32_t scale_a_inner, int32_t scale_a_outer,
    int32_t scale_b_inner, int32_t scale_b_outer,
    int64_t* __restrict__ scalePtrA_out, int64_t* __restrict__ scalePtrB_out) {
  int i = threadIdx.x;

  if (i == 0) {
    *alpha_ptr = 1.0f;
    *beta_ptr = 0.0f;
  }

  int32_t delta = 0;
  int64_t group_start = 0;
  if (offs != nullptr) {
    int32_t end = offs[i];
    int32_t start_val = (i == 0) ? 0 : offs[i - 1];
    delta = end - start_val;
    group_start = static_cast<int64_t>(start_val);
  }

  m_out[i] = m_is_delta ? delta : cublas_m;
  n_out[i] = n_is_delta ? delta : cublas_n;
  k_out[i] = k_is_delta ? delta : cublas_k;

  lda_out[i] = lda_val;
  ldb_out[i] = ldb_val;
  ldd_out[i] = ldd_val;

  APtr_out[i] = base_A + group_start * a_offs_stride + i * a_idx_stride;
  BPtr_out[i] = base_B + group_start * b_offs_stride + i * b_idx_stride;
  DPtr_out[i] = base_D + group_start * d_offs_stride + i * d_idx_stride;

  alphaPtr_out[i] = reinterpret_cast<int64_t>(alpha_ptr);
  betaPtr_out[i] = reinterpret_cast<int64_t>(beta_ptr);

  if (scalePtrA_out != nullptr) {
    if (scale_a_stride_bytes != 0) {
      // Uniform stride (3D/3D or GroupWise)
      scalePtrA_out[i] = base_scale_a + i * scale_a_stride_bytes;
    } else {
      // Variable-size groups: prefix-sum over per-group scale sizes.
      // A 0 value in scale_a_inner or scale_a_outer means "use delta from offs".
      int64_t offset = 0;
      for (int j = 0; j < i; j++) {
        int32_t dim_j = (j == 0) ? offs[j] : offs[j] - offs[j - 1];
        int32_t inner = scale_a_inner ? scale_a_inner : dim_j;
        int32_t outer = scale_a_outer ? scale_a_outer : dim_j;
        offset += cublas_vec32_scale_size(inner, outer);
      }
      scalePtrA_out[i] = base_scale_a + offset;
    }
  }
  if (scalePtrB_out != nullptr) {
    if (scale_b_stride_bytes != 0) {
      scalePtrB_out[i] = base_scale_b + i * scale_b_stride_bytes;
    } else {
      int64_t offset = 0;
      for (int j = 0; j < i; j++) {
        int32_t dim_j = (j == 0) ? offs[j] : offs[j] - offs[j - 1];
        int32_t inner = scale_b_inner ? scale_b_inner : dim_j;
        int32_t outer = scale_b_outer ? scale_b_outer : dim_j;
        offset += cublas_vec32_scale_size(inner, outer);
      }
      scalePtrB_out[i] = base_scale_b + offset;
    }
  }
}

void launch_populate_cublas_grouped_args(
    int batchCount,
    const int32_t* offs,
    int64_t base_A, int64_t base_B, int64_t base_D,
    int32_t cublas_m, int32_t cublas_n, int32_t cublas_k,
    bool m_is_delta, bool n_is_delta, bool k_is_delta,
    int32_t lda_val, int32_t ldb_val, int32_t ldd_val,
    int64_t a_offs_stride, int64_t a_idx_stride,
    int64_t b_offs_stride, int64_t b_idx_stride,
    int64_t d_offs_stride, int64_t d_idx_stride,
    int32_t* m_out, int32_t* n_out, int32_t* k_out,
    int32_t* lda_out, int32_t* ldb_out, int32_t* ldd_out,
    int64_t* APtr_out, int64_t* BPtr_out, int64_t* DPtr_out,
    int64_t* alphaPtr_out, int64_t* betaPtr_out,
    float* alpha_ptr, float* beta_ptr,
    int64_t base_scale_a, int64_t base_scale_b,
    int64_t scale_a_stride_bytes, int64_t scale_b_stride_bytes,
    int32_t scale_a_inner, int32_t scale_a_outer,
    int32_t scale_b_inner, int32_t scale_b_outer,
    int64_t* scalePtrA_out, int64_t* scalePtrB_out,
    cudaStream_t stream) {
  TORCH_CHECK(batchCount > 0 && batchCount <= 1024,
      "batchCount must be in [1, 1024], got ", batchCount);
  populate_cublas_grouped_args_kernel<<<1, batchCount, 0, stream>>>(
      offs, base_A, base_B, base_D,
      cublas_m, cublas_n, cublas_k,
      m_is_delta, n_is_delta, k_is_delta,
      lda_val, ldb_val, ldd_val,
      a_offs_stride, a_idx_stride,
      b_offs_stride, b_idx_stride,
      d_offs_stride, d_idx_stride,
      m_out, n_out, k_out,
      lda_out, ldb_out, ldd_out,
      APtr_out, BPtr_out, DPtr_out,
      alphaPtr_out, betaPtr_out,
      alpha_ptr, beta_ptr,
      base_scale_a, base_scale_b,
      scale_a_stride_bytes, scale_b_stride_bytes,
      scale_a_inner, scale_a_outer,
      scale_b_inner, scale_b_outer,
      scalePtrA_out, scalePtrB_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace at::native
