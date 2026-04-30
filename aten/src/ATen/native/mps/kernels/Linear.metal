// MPP (MetalPerformancePrimitives) matmul2d kernel for F.linear.
// Computes C[M,N] = A[M,K] @ B[N,K]^T with optional fused bias.
// Multiple tile-size variants are instantiated; the host selects the best one
// based on input dimensions.
// Requires Metal 4.0 (macOS 26+).
#if __METAL_VERSION__ >= 400
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename T, bool HAS_BIAS, int TILE_M, int TILE_N>
kernel void mpp_linear(
    device T* A [[buffer(0)]],
    device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    device T* bias [[buffer(3)]],
    constant uint3 &sizes [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint M = sizes.x;
    const uint K = sizes.y;
    const uint N = sizes.z;

    device T* A_tile = A + tgid.y * TILE_M * K;
    device T* B_tile = B + tgid.x * TILE_N * K;
    device T* C_tile = C + tgid.y * TILE_M * N + tgid.x * TILE_N;

    using ext_a = extents<int32_t, dynamic_extent, TILE_M>;
    using ext_b = extents<int32_t, dynamic_extent, TILE_N>;
    using ext_c = extents<int32_t, TILE_N, TILE_M>;

    tensor<device T, ext_a, tensor_inline> mA(A_tile, ext_a(K));
    tensor<device T, ext_b, tensor_inline> mB(B_tile, ext_b(K));
    tensor<device T, ext_c, tensor_inline> mC(
        C_tile, ext_c(), array<int32_t, 2>{1, (int)N});

    constexpr auto desc = matmul2d_descriptor(
        TILE_M, TILE_N, static_cast<int>(dynamic_extent), false, true);
    matmul2d<desc, execution_simdgroups<4>> op;
    op.run(mA, mB, mC);

    if (HAS_BIAS) {
        constexpr uint TG_SIZE = 4 * 32;
        device T* bias_tile = bias + tgid.x * TILE_N;
        for (uint i = tid; i < TILE_M * TILE_N; i += TG_SIZE) {
            uint m = i / TILE_N;
            uint n = i % TILE_N;
            C_tile[m * N + n] += bias_tile[n];
        }
    }
}

#define INSTANTIATE_TILE(T, suffix, TM, TN)                                       \
template [[host_name("mpp_linear_" #TM "x" #TN "_" #suffix)]]                    \
kernel void mpp_linear<T, false, TM, TN>(device T*, device T*, device T*,        \
    device T*, constant uint3&, uint2, uint);                                     \
template [[host_name("mpp_linear_bias_" #TM "x" #TN "_" #suffix)]]               \
kernel void mpp_linear<T, true, TM, TN>(device T*, device T*, device T*,         \
    device T*, constant uint3&, uint2, uint);

#define INSTANTIATE(T, suffix)     \
INSTANTIATE_TILE(T, suffix, 32, 32)  \
INSTANTIATE_TILE(T, suffix, 64, 64)  \
INSTANTIATE_TILE(T, suffix, 128, 64)

INSTANTIATE(float, float)
INSTANTIATE(half, half)
INSTANTIATE(bfloat, bfloat)
#undef INSTANTIATE
#undef INSTANTIATE_TILE
#endif // __METAL_VERSION__ >= 400
