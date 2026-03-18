#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace at::native {

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
    int64_t* scalePtrA_out, int64_t* scalePtrB_out,
    cudaStream_t stream);

} // namespace at::native
