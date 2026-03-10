#include <ATen/native/cuda/CublasGroupedArgs.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

namespace at::native {

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
    float* __restrict__ alpha_ptr, float* __restrict__ beta_ptr) {
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
      alpha_ptr, beta_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace at::native
