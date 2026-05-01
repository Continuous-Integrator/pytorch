//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/SoftMaxKernel.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SoftMaxKernel_metallib.h>
#endif

static bool canUseMetalSoftmax(const Tensor& input, int64_t dim) {
  int64_t ndim = input.dim();
  if (ndim == 0)
    return false;
  int64_t wrapped = maybe_wrap_dim(dim, ndim);
  return wrapped == ndim - 1;
}

static SoftmaxParams makeForwardParams(const Tensor& input, int64_t dim) {
  SoftmaxParams params = {};
  int64_t ndim = input.dim();
  params.axis_size = static_cast<uint32_t>(input.size(dim));
  params.stride_a = static_cast<uint32_t>(input.stride(dim));
  params.stride_b = 0;
  params.ndim = static_cast<uint32_t>(ndim);
  int outer_idx = 0;
  for (int64_t d = 0; d < ndim; d++) {
    if (d == dim)
      continue;
    params.outer_sizes[outer_idx] = static_cast<uint32_t>(input.size(d));
    params.outer_strides_a[outer_idx] = static_cast<uint32_t>(input.stride(d));
    params.outer_strides_b[outer_idx] = 0;
    outer_idx++;
  }
  return params;
}

static SoftmaxParams makeBackwardParams(const Tensor& grad, const Tensor& output, int64_t dim) {
  SoftmaxParams params = {};
  int64_t ndim = grad.dim();
  params.axis_size = static_cast<uint32_t>(grad.size(dim));
  params.stride_a = static_cast<uint32_t>(grad.stride(dim));
  params.stride_b = static_cast<uint32_t>(output.stride(dim));
  params.ndim = static_cast<uint32_t>(ndim);
  int outer_idx = 0;
  for (int64_t d = 0; d < ndim; d++) {
    if (d == dim)
      continue;
    params.outer_sizes[outer_idx] = static_cast<uint32_t>(grad.size(d));
    params.outer_strides_a[outer_idx] = static_cast<uint32_t>(grad.stride(d));
    params.outer_strides_b[outer_idx] = static_cast<uint32_t>(output.stride(d));
    outer_idx++;
  }
  return params;
}

} // namespace mps

static void get_shapes(MPSShape* input_shape_readonly,
                       NSMutableArray<NSNumber*>*& input_shape,
                       int num_input_dims,
                       c10::MemoryFormat memory_format) {
  if (memory_format == at::MemoryFormat::Contiguous) {
    for (int i = 0; i < num_input_dims; i++)
      input_shape[i] = input_shape_readonly[i];
  } else { // ChannelsLast
    auto num_channels = input_shape_readonly[1];
    input_shape[0] = input_shape_readonly[0];
    for (int i = 1; i < num_input_dims - 1; i++)
      input_shape[i] = input_shape_readonly[i + 1];
    input_shape[num_input_dims - 1] = num_channels;
  }
}

TORCH_IMPL_FUNC(softmax_mps_out)
(const Tensor& input_, const int64_t dim, const bool half_to_float, const Tensor& output) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on MPS");
  TORCH_CHECK(c10::isFloatingType(input_.scalar_type()), "softmax only supported for floating types");

  if (input_.numel() == 0) {
    return;
  }

  Tensor input;
  if (input_.dim() == 0) {
    input = input_.view(1);
  } else
    input = input_;

  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < input.dim(), "Softmax:dim must be non-negative and less than input dimensions");

  if (mps::canUseMetalSoftmax(input, dim_)) {
    using namespace mps;
    MPSStream* stream = getCurrentMPSStream();

    int64_t axis_size = input.size(dim_);
    int64_t outer_size = input.numel() / axis_size;
    auto params = makeForwardParams(input, dim_);

    @autoreleasepool {
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        const int N_READS = 4;
        auto metalType = mps::scalarToMetalTypeString(input);
        id<MTLComputePipelineState> kernel;
        if (axis_size <= 1024 * N_READS) {
          kernel = mps::lib.getPipelineStateForFunc("softmax_forward_single_row_" + metalType);
        } else {
          kernel = mps::lib.getPipelineStateForFunc("softmax_forward_looped_" + metalType);
        }

        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:kernel];

        mps::mtl_setArgs(encoder, input, output, params);

        MTLSize threadsPerGroup = MTLSizeMake(
            std::min(static_cast<int64_t>((axis_size + N_READS - 1) / N_READS), static_cast<int64_t>(1024)), 1, 1);
        MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      });
    }
    return;
  }

  // MPSGraph fallback for non-last-dim softmax
  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);

  const auto memory_format = input.suggest_memory_format();

  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string mem_format_key = get_mem_format_string(memory_format);
    MPSShape* input_shape_readonly = mps::getMPSShape(input);
    int num_input_dims = [input_shape_readonly count];
    TORCH_CHECK(memory_format != at::MemoryFormat::ChannelsLast || num_input_dims == 4,
                "ChannelsLast implies 4d tensor")
    NSMutableArray<NSNumber*>* input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

    get_shapes(input_shape_readonly, input_shape, num_input_dims, memory_format);

    if (memory_format == at::MemoryFormat::ChannelsLast && dim_ > 0 && !is_macOS_15_0_or_newer) {
      switch (dim_) {
        case 1:
          dim_ = 3;
          break;
        case 2:
          dim_ = 1;
          break;
        case 3:
          dim_ = 2;
          break;
        default:
          assert(0 && "Invalid dim\n");
      }
    }

    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    std::string key = "softmax_mps_out" + getTensorsStringKey(input, true, /*exclude_shape*/ true) + ":" +
        mem_format_key + ":" + std::to_string(dim_);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()));

      MPSGraphTensor* outputTensor = [mpsGraph softMaxWithTensor:inputTensor axis:(NSInteger)dim_ name:nil];

      if (memory_format == at::MemoryFormat::ChannelsLast && !is_macOS_15_0_or_newer) {
        auto N = input_shape[0];
        auto H = input_shape[1];
        auto W = input_shape[2];
        auto C = input_shape[3];

        outputTensor = [mpsGraph reshapeTensor:outputTensor
                                     withShape:@[ N, ([NSNumber numberWithInt:[H intValue] * [W intValue]]), C ]
                                          name:nil];
        outputTensor = [mpsGraph transposeTensor:outputTensor dimension:1 withDimension:2 name:nil];
        outputTensor = [mpsGraph reshapeTensor:outputTensor withShape:@[ N, C, H, W ] name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder =
        Placeholder(cachedGraph->inputTensor_, input, is_macOS_15_0_or_newer ? nil : input_shape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(softmax_backward_mps_out)
(const Tensor& grad_, const Tensor& output_, int64_t dim, ScalarType input_dtype, const Tensor& grad_input) {
  if (output_.numel() == 0) {
    return;
  }

  Tensor grad;
  if (grad_.dim() == 0) {
    grad = grad_.view(1);
  } else
    grad = grad_;

  Tensor output;
  if (output_.dim() == 0) {
    output = output_.view(1);
  } else
    output = output_;

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < grad.dim(), "Grad:dim must be non-negative and less than input dimensions");

  if (mps::canUseMetalSoftmax(output, dim_) && mps::canUseMetalSoftmax(grad, dim_)) {
    using namespace mps;
    MPSStream* stream = getCurrentMPSStream();

    const int N_READS = 4;
    int64_t axis_size = output.size(dim_);
    int64_t outer_size = output.numel() / axis_size;
    int64_t tg_size = std::min(static_cast<int64_t>((axis_size + N_READS - 1) / N_READS), static_cast<int64_t>(1024));
    auto params = makeBackwardParams(grad, output, dim_);

    // Split backward across threadgroups when few rows leave the GPU underutilized.
    // Empirically, outer_size >= 4 already saturates the GPU for looped axis sizes —
    // the barrier overhead of two-pass then exceeds its occupancy benefit. Below 4 rows,
    // two-pass meaningfully improves parallelism. The max_chunks >= 2 guard ensures there
    // is actually work to split.
    constexpr int64_t kMinOccupancyTG = 4;
    int64_t elems_per_tg = tg_size * N_READS;
    int64_t max_chunks = axis_size / elems_per_tg;
    bool use_two_pass = (axis_size > elems_per_tg) && (outer_size < kMinOccupancyTG) && (max_chunks >= 2);

    Tensor partial_sums;
    if (use_two_pass) {
      params.num_chunks = static_cast<uint32_t>(max_chunks);
      partial_sums = at::empty({outer_size * max_chunks}, grad.options().dtype(at::kFloat));
    }

    @autoreleasepool {
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        auto metalType = mps::scalarToMetalTypeString(output);
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);

        if (use_two_pass) {
          auto dot_kernel = mps::lib.getPipelineStateForFunc("softmax_backward_2pass_dot_" + metalType);
          [encoder setComputePipelineState:dot_kernel];
          mps::mtl_setArgs(encoder, grad, output, partial_sums, params);
          MTLSize numGroups = MTLSizeMake(static_cast<NSUInteger>(params.num_chunks) * outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

          auto grad_kernel = mps::lib.getPipelineStateForFunc("softmax_backward_2pass_grad_" + metalType);
          [encoder setComputePipelineState:grad_kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input, partial_sums, params);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        } else {
          id<MTLComputePipelineState> kernel;
          if (axis_size <= 1024 * N_READS) {
            kernel = mps::lib.getPipelineStateForFunc("softmax_backward_single_row_" + metalType);
          } else {
            kernel = mps::lib.getPipelineStateForFunc("softmax_backward_looped_" + metalType);
          }

          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input, params);
          MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }
      });
    }
    return;
  }

  // MPSGraph fallback for non-last-dim backward
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* grad_shape = mps::getMPSShape(grad);
    NSString* ns_shape_key = [[grad_shape valueForKey:@"description"] componentsJoinedByString:@","];

    std::string key = "softmax_backward_mps_out:" + getMPSTypeString(output) + ":" + [ns_shape_key UTF8String] + ":" +
        std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* softmaxTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(output), grad_shape);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad), grad_shape);

      MPSGraphTensor* mulTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                            secondaryTensor:gradOutputTensor
                                                                       name:nil];
      MPSGraphTensor* mulSumTensor = [mpsGraph reductionSumWithTensor:mulTensor axis:(NSInteger)dim_ name:nil];
      MPSGraphTensor* gradSubTensor = [mpsGraph subtractionWithPrimaryTensor:gradOutputTensor
                                                             secondaryTensor:mulSumTensor
                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                                  secondaryTensor:gradSubTensor
                                                                             name:nil];

      newCachedGraph->outputTensor_ = softmaxTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder softmaxPlaceholder = Placeholder(cachedGraph->outputTensor_, output, grad_shape);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad, grad_shape);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(softmaxPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

} // namespace at::native
