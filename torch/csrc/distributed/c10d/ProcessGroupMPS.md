# ProcessGroupMPS: Supported Operations

RDMA-only c10d backend for Apple Silicon Macs connected via Thunderbolt 5.
Collectives are layered on top of [JACCL](../../../../../third_party/jaccl/),
MLX's standalone RDMA-over-Thunderbolt library (vendored in
`third_party/jaccl/`). Construction requires `ibv_alloc_pd` to succeed on every
rank; otherwise the constructor throws and users should select the `gloo`
backend.

## Implemented

| Op          | JACCL primitive                      | Notes                                                                              |
|-------------|--------------------------------------|------------------------------------------------------------------------------------|
| `allreduce` | `all_sum` / `all_min` / `all_max`    | `SUM` / `MIN` / `MAX` only. Dtypes: any JACCL dtype (fp16/bf16/fp32/fp64/int8-64/uint8-64/bool/complex64) |
| `broadcast` | `send` + `recv`                      | Root sends to each peer; all ranks, any root                                       |
| `send`      | `send`                               | Single tensor, any dst                                                             |
| `recv`      | `recv`                               | Single tensor, any src                                                             |
| `barrier`   | 1-byte `all_sum`                     | JACCL has no dedicated barrier; a minimal all-reduce synchronises with the same fences |

## Not implemented

| Op                                      | Needed for                                            | Rough difficulty                                                     |
|-----------------------------------------|-------------------------------------------------------|----------------------------------------------------------------------|
| `allreduce` with `PRODUCT`              | Uncommon reductions                                   | Skip - JACCL does not expose a product reduction                     |
| `allgather` / `_allgather_base`         | `DistributedDataParallel` init, model parallel, FSDP  | Trivial - forward to `jaccl::Group::all_gather`                      |
| `reduce_scatter` / `_reduce_scatter_base` | FSDP, ZeRO-2/3, tensor parallel                     | Medium - compose all_sum + per-chunk slicing                         |
| `reduce` (to single rank)               | Some MPI-style workloads                              | Easy - allreduce then discard on non-root                            |
| `gather` (to single rank)               | Rare                                                  | Easy - allgather and discard, or dedicated send-to-root              |
| `scatter` (from single rank)            | Rare                                                  | Easy - root sends slice to each peer                                 |
| `alltoall` / `alltoall_base`            | MoE, sequence-parallel                                | Medium - N-way permutation                                           |
| `*_coalesced` variants                  | Perf; fallback is loop-of-single                      | Trivial loop fallback until it matters                               |
| `allgather_into_tensor_coalesced`       | DDP gradient bucketization                            | Trivial fallback                                                     |
| `recvAnysource`                         | Arbitrary peer recv                                   | Skip - UC connections don't help here                                |
| `allreduce_sparse`                      | Sparse gradients                                      | Skip unless requested                                                |
| `startCoalescing` / `endCoalescing`     | Batched collectives                                   | Skip for now                                                         |
| `monitored_barrier`                     | Timeout-aware barrier                                 | Skip for now                                                         |

## Priority order

1. **`allgather`** unblocks `DistributedDataParallel` (currently errors at
   init with `Backend mps does not support allgather`). One-line forward to
   `jaccl::Group::all_gather`.
2. **`reduce_scatter`** unblocks FSDP, the default training setup for
   memory-constrained multi-Mac runs.
3. Everything else is either trivially composable from existing
   primitives or rarely used enough that "not implemented" is acceptable.
