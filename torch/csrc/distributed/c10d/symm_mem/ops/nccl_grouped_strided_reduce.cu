#include <c10/cuda/CUDAGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

// Simultaneously reduce N strided 2-D tensors, routing each to a specific
// destination rank (dst_ranks[i]).  Only the destination rank writes the
// reduced value; all other ranks participate in the LSA barrier but do no
// writes.
//
// Ownership must be balanced: every rank must own the same number of tensors
// (N % world_size == 0 and dst_ranks distributes evenly).
//
// Example use case: EP+Muon "grouped reduce", where each tensor is an expert
// gradient column-block from a Grouped GEMM buffer.

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

#ifdef NCCL_DEVICE_HAS_REDUCE_COPY

#define NCCL_GR_MAX_INPUTS 64
#define NCCL_GR_MAX_CTAS_PER_INPUT 16
#define NCCL_GR_THREADS_PER_CTA 128
#define NCCL_GR_MAX_CTA_COUNT (NCCL_GR_MAX_INPUTS * NCCL_GR_MAX_CTAS_PER_INPUT)

// Arrays are indexed by owned slot (0..n_owned-1), not by global input index.
struct NcclGroupedStridedReduceInfo {
  size_t byte_offsets[NCCL_GR_MAX_INPUTS];
  // Write destination: either user-provided out tensor or the input itself.
  void* dst_ptrs[NCCL_GR_MAX_INPUTS];
  // Row stride (in elements) for dst_ptrs: cols if contiguous out, outer_stride otherwise.
  int64_t dst_strides[NCCL_GR_MAX_INPUTS];
};

template <typename T>
__global__ void nccl_grouped_strided_reduce_kernel(
    ncclWindow_t window,
    NcclGroupedStridedReduceInfo info,
    int rows,
    int cols,
    int64_t outer_stride,
    int blocks_per_input,
    ncclDevComm devComm) {
  const int slot = blockIdx.x / blocks_per_input;
  const int local_block = blockIdx.x % blocks_per_input;

  // One dedicated LSA barrier per CTA (index = blockIdx.x).
  // All ranks must participate in both syncs regardless of whether they do work.
  ncclLsaBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(),
      devComm,
      ncclTeamLsa(devComm),
      devComm.lsaBarrier,
      blockIdx.x};
  // Acquire: wait for all ranks to have their data ready in the window.
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  // Distribute rows across the blocks_per_input CTAs for this input.
  // Within each row, ncclLsaReduceSum reads from all peers with vectorized
  // (packed) loads, reduces, and writes to dst — no manual peer loop needed.
  const size_t base_byte_offset = info.byte_offsets[slot];
  T* dst_base = reinterpret_cast<T*>(info.dst_ptrs[slot]);

  for (int row = local_block; row < rows; row += blocks_per_input) {
    const size_t row_offset =
        base_byte_offset +
        static_cast<size_t>(row * outer_stride) * sizeof(T);
    T* dst_row = dst_base + row * info.dst_strides[slot];
    ncclLsaReduceSum(ncclCoopCta(), window, row_offset, dst_row, cols, devComm);
  }

  // Release: signal to all peers that we are done accessing window memory.
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

#endif // NCCL_DEVICE_HAS_REDUCE_COPY

void nccl_grouped_strided_reduce(
    at::TensorList inputs,
    at::IntArrayRef dst_ranks,
    const std::string& group_name,
    at::TensorList out) {
#ifdef NCCL_DEVICE_HAS_REDUCE_COPY
  const int n_inputs = static_cast<int>(inputs.size());
  TORCH_CHECK(
      n_inputs > 0, "nccl_grouped_strided_reduce: inputs must be non-empty");
  TORCH_CHECK(
      n_inputs <= NCCL_GR_MAX_INPUTS,
      "nccl_grouped_strided_reduce: too many inputs: ",
      n_inputs,
      " (max ",
      NCCL_GR_MAX_INPUTS,
      ")");
  TORCH_CHECK(
      static_cast<int>(dst_ranks.size()) == n_inputs,
      "nccl_grouped_strided_reduce: dst_ranks.size() must match inputs.size()");

  const auto& t0 = inputs[0];
  TORCH_CHECK(t0.dim() == 2, "nccl_grouped_strided_reduce: tensors must be 2-D");
  TORCH_CHECK(
      t0.stride(-1) == 1,
      "nccl_grouped_strided_reduce: innermost dimension must be contiguous "
      "(stride[-1] == 1)");

  for (int i = 1; i < n_inputs; i++) {
    TORCH_CHECK(
        inputs[i].sizes() == t0.sizes() &&
            inputs[i].stride(0) == t0.stride(0) &&
            inputs[i].stride(-1) == 1,
        "nccl_grouped_strided_reduce: all tensors must have the same shape and outer stride");
    TORCH_CHECK(
        inputs[i].device() == t0.device(),
        "nccl_grouped_strided_reduce: all tensors must be on the same device");
    TORCH_CHECK(
        inputs[i].scalar_type() == t0.scalar_type(),
        "nccl_grouped_strided_reduce: all tensors must have the same dtype");
  }

  auto symm_mem = c10d::symmetric_memory::rendezvous(t0, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "nccl_grouped_strided_reduce: tensor must be allocated via NCCL symmetric "
      "memory (use empty_strided_p2p with NCCL backend)");

  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(
      nccl_hdl != nullptr,
      "nccl_grouped_strided_reduce: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(t0.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = t0.device();

  auto& manager = c10d::symmetric_memory::NCCLDevCommManager::get(device);
  ncclComm_t comm = manager.get_comm(group_name);

  static constexpr char kDevcommKey[] = "nccl_grouped_strided_reduce";
  auto devcomm_opt = manager.get_devcomm(group_name, kDevcommKey);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = NCCL_GR_MAX_CTA_COUNT;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_grouped_strided_reduce");
    devcomm_opt = manager.register_devcomm(group_name, devcomm, kDevcommKey);
  }
  ncclDevComm& devcomm = devcomm_opt->get();

  const int my_rank = devcomm.rank;
  const int world_size = devcomm.nRanks;

  // Count owned inputs and verify balanced distribution.
  int n_owned = 0;
  for (int i = 0; i < n_inputs; i++) {
    n_owned += (static_cast<int>(dst_ranks[i]) == my_rank ? 1 : 0);
  }
  TORCH_CHECK(
      n_owned * world_size == n_inputs,
      "nccl_grouped_strided_reduce: dst_ranks must distribute tensors evenly across "
      "all ranks (this rank owns ",
      n_owned,
      " of ",
      n_inputs,
      " tensors, world_size=",
      world_size,
      ")");

  if (!out.empty()) {
    TORCH_CHECK(
        static_cast<int>(out.size()) == n_owned,
        "nccl_grouped_strided_reduce: out.size() (",
        out.size(),
        ") must equal the number of tensors owned by this rank (",
        n_owned,
        ")");
    for (int j = 0; j < static_cast<int>(out.size()); j++) {
      TORCH_CHECK(
          out[j].sizes() == t0.sizes(),
          "nccl_grouped_strided_reduce: out[",
          j,
          "] must have the same shape as inputs");
      TORCH_CHECK(
          out[j].is_contiguous(),
          "nccl_grouped_strided_reduce: out[",
          j,
          "] must be contiguous");
      TORCH_CHECK(
          out[j].scalar_type() == t0.scalar_type(),
          "nccl_grouped_strided_reduce: out[",
          j,
          "] must have the same dtype as inputs");
    }
  }

  const int rows = static_cast<int>(t0.size(0));
  const int cols = static_cast<int>(t0.size(1));
  const int64_t outer_stride = t0.stride(0);
  const int numel = rows * cols;
  // ncclLsaReduceSum default UNROLL=4*16/sizeof(T): each thread covers
  // UNROLL elements per call, so one CTA covers elems_per_cta elements.
  const int unroll = 4 * 16 / static_cast<int>(t0.element_size());
  const int elems_per_cta = NCCL_GR_THREADS_PER_CTA * unroll;
  const int ctas_per_input = std::max(1, std::min(
      (numel + elems_per_cta - 1) / elems_per_cta,
      NCCL_GR_MAX_CTAS_PER_INPUT));
  const size_t window_base_offset = nccl_hdl->get_offset();

  // Populate info indexed by owned slot.
  NcclGroupedStridedReduceInfo info;
  int slot = 0;
  for (int i = 0; i < n_inputs; i++) {
    if (static_cast<int>(dst_ranks[i]) != my_rank) {
      continue;
    }
    info.byte_offsets[slot] =
        window_base_offset +
        static_cast<size_t>(inputs[i].storage_offset()) *
            inputs[i].element_size();
    if (out.empty()) {
      info.dst_ptrs[slot] = inputs[i].data_ptr();
      info.dst_strides[slot] = outer_stride;
    } else {
      info.dst_ptrs[slot] = out[slot].data_ptr();
      info.dst_strides[slot] = cols;
    }
    slot++;
  }

  auto window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_grouped_strided_reduce: NCCL window is null");

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      t0.scalar_type(),
      "nccl_grouped_strided_reduce",
      [&]() {
        nccl_grouped_strided_reduce_kernel<scalar_t>
            <<<n_owned * ctas_per_input, NCCL_GR_THREADS_PER_CTA, 0, stream>>>(
                window,
                info,
                rows,
                cols,
                outer_stride,
                ctas_per_input,
                devcomm);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
#else
  TORCH_CHECK(
      false,
      "nccl_grouped_strided_reduce requires NCCL >= 2.29.7 with reduce copy support");
#endif // NCCL_DEVICE_HAS_REDUCE_COPY
}

} // namespace c10d::nccl_extension
