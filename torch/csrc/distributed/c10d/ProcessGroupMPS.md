# ProcessGroupMPS: Supported Operations

RDMA-only c10d backend for Apple Silicon Macs connected via Thunderbolt 5.
Construction requires `ibv_alloc_pd` to succeed on every rank; otherwise
the constructor throws and users should select the `gloo` backend.

## Implemented

| Op          | Transport                       | Notes                                                                      |
|-------------|---------------------------------|----------------------------------------------------------------------------|
| `allreduce` | RDMA mesh                       | `SUM` / `PRODUCT` / `MIN` / `MAX` over `float32` / `float64` / `int32` / `int64` |
| `broadcast` | RDMA mesh (root sends to peers) | All ranks, any root                                                        |
| `send`      | RDMA point-to-point             | Single tensor, any dst                                                     |
| `recv`      | RDMA point-to-point             | Single tensor, any src                                                     |
| `barrier`   | TCP side-channel allgather      | No RDMA needed                                                             |

## Not implemented

| Op                                      | Needed for                                                      | Rough difficulty                                                                  |
|-----------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `allgather` / `_allgather_base`         | `DistributedDataParallel` init, model parallel, FSDP            | Easy - MLX has `mesh.allGather`, port + dispatch                                  |
| `reduce_scatter` / `_reduce_scatter_base` | FSDP, ZeRO-2/3, tensor parallel                                | Medium - combine reduce + per-chunk partitioning                                  |
| `reduce` (to single rank)               | Some MPI-style workloads                                        | Easy - allreduce then discard on non-root                                         |
| `gather` (to single rank)               | Rare                                                            | Easy - allgather and discard, or dedicated send-to-root                           |
| `scatter` (from single rank)            | Rare                                                            | Easy - root sends slice to each peer                                              |
| `alltoall` / `alltoall_base`            | MoE, sequence-parallel                                          | Medium - N-way permutation                                                        |
| `*_coalesced` variants                  | Perf optimization; fallback is loop-of-single                   | Trivial loop fallback until it matters                                            |
| `allgather_into_tensor_coalesced`       | DDP gradient bucketization                                      | Trivial fallback                                                                  |
| `recvAnysource`                         | Arbitrary peer recv                                             | Skip - UC connections don't help here                                             |
| `allreduce_sparse`                      | Sparse gradients                                                | Skip unless requested                                                             |
| `startCoalescing` / `endCoalescing`     | Batched collectives                                             | Skip for now                                                                      |
| `monitored_barrier`                     | Timeout-aware barrier                                           | Skip for now                                                                      |

## Priority order

1. **`allgather`** unblocks `DistributedDataParallel` (currently errors at
   init with `Backend mps does not support allgather`).
2. **`reduce_scatter`** unblocks FSDP, the default training setup for
   memory-constrained multi-Mac runs.
3. Everything else is either trivially composable from existing
   primitives or rarely used enough that "not implemented" is acceptable.
