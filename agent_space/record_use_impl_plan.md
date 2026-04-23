# Implementation plan: `record_use` (FIFO-chain variant)

Companion to `record_use_design_doc.md`. Goal: rework PR #181189 from the earlier event-polling prototype to the FIFO-chain mechanism — issue `cudaStreamWaitEvent` on the requesting stream at allocation time, no `cudaEventQuery` polling on the critical path.

## What to keep from the current prototype

- `Block::use_events` field (rename to `pending_waits` — see Step 1).
- `DeviceCachingAllocator::recordUse(block, stream)`: records event eagerly, pushes onto the block's list.
- `CUDAAllocator::recordUse` / `DeviceAllocator::recordUse` virtuals + default delegating to `recordStream`.
- ATen: `record_use` entry in `native_functions.yaml` + `RecordUse.cu` kernel.
- Python: `_tensor_docs.py`, `overrides.py`, `_dynamo/variables/{streams,tensor}.py`, `nested/_internal/ops.py`.
- Codegen allow-list entries (`FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT`, `MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION`, `gen_variable_type.py`).
- `std::move` fix in `get_free_block_from_mempool`'s `params = std::move(mempool_params)`.

## What to remove

- `consume_use_events(block)` helper.
- The `consume_use_events(block)` call in `free()`.
- The `free()` gate extension `!stream_uses.empty() || !use_events.empty()` — revert to `!stream_uses.empty()`.
- Assertion extensions in `free_block()` and `release_free_blocks_in_capture` that check `use_events.empty()` — revert.

## What to add

- An `issue_pending_waits(block, requesting_stream)` helper, called at the allocation hand-off point.
- One call site at the correct spot in the allocation path.

## Steps (ordered, each reviewable)

### Step 1 — Rename `use_events` → `pending_waits`

Optional but recommended. `stream_uses` already means something different (imprecise, record_stream path); `pending_waits` reads as "wait_events we owe the reusing stream." Applies to `Block`, `DeviceCachingAllocator::recordUse`, and any internal references.

### Step 2 — Revert the free-path integration

In `c10/cuda/CUDACachingAllocator.cpp`:

- Delete `consume_use_events`.
- In `free()`, revert the gate to the original `if (!block->stream_uses.empty())` form.
- In `free_block()`, remove `&& block->use_events.empty()` from the internal assert.
- In `release_free_blocks_in_capture` and `insert_events_deferred_until_no_capture`, remove the `TORCH_INTERNAL_ASSERT(block->use_events.empty())` lines.

After this step, blocks return to the pool immediately on free (subject to the unchanged `record_stream` path), with `pending_waits` still attached.

### Step 3 — Add `issue_pending_waits`

```cpp
// Called with allocator mutex held.
void issue_pending_waits(Block* block, cuda::CUDAStream requesting_stream) {
  if (block->pending_waits.empty()) return;
  c10::DeviceIndex prev_device = 0;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&prev_device));
  C10_CUDA_CHECK(c10::cuda::SetDevice(requesting_stream.device_index()));
  for (auto& [event, recorded_stream] : block->pending_waits) {
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(requesting_stream.stream(), *event, 0));
  }
  C10_CUDA_CHECK(c10::cuda::MaybeSetDevice(prev_device));
  block->pending_waits.clear();
}
```

### Step 4 — Wire `issue_pending_waits` into the allocation hand-off

The exact site: `DeviceCachingAllocator::alloc_found_block` (where a block has been selected — via `get_free_block`, free-memory callback retry, or fresh segment — and is about to be returned). The requesting stream is `params.stream()`. Call `issue_pending_waits(block, cuda::CUDAStream(...params.stream()...))` right after `block->allocated = true` is set, before returning the block pointer.

If `alloc_found_block` doesn't hold the mutex directly, verify locking; the helper assumes mutex held. The existing `malloc` path already holds the mutex through block selection.

### Step 5 — Block splitting policy

In `get_free_block`, when a block is split:

- The child returned to the caller inherits `pending_waits`.
- The remainder inserted back into the pool has **empty** `pending_waits`.

Rationale (monotone-fire): once `issue_pending_waits` runs on the returned child, the reusing stream's FIFO orders the physical memory as safe. The event has fired (or is about to); any subsequent allocation of the other half on any stream is automatically safe on that physical memory. See design doc §"Interaction with expandable segments / block splitting."

Concrete: in the split path, `std::move` the `pending_waits` vector onto the returned block, leaving the remainder's vector empty.

### Step 6 — Same-device assumption

`recordUse(block, stream)`: if `stream.device_index() != block->device`, for PR 1, **skip** (no-op + warn). Cross-device wait_events are valid CUDA but depend on peer-access; scope out for PR 1. Follow-up can lift this.

### Step 7 — Graph capture fallback

If `captures_underway` is non-empty, `recordUse` becomes a no-op + `TORCH_WARN_ONCE`. FSDP2 also doesn't use this pattern under capture. A precise-in-capture story is a separate RFC.

### Step 8 — Replace the tests

Delete the 5 tests added in the current prototype. Write new tests in `test/test_cuda.py`:

| # | Test | What it verifies |
|---|---|---|
| 1 | `test_record_use_basic` | Alloc on A; `record_use(B)`; del; alloc on A → issued `cudaStreamWaitEvent` before return. No polling on the path. |
| 2 | `test_record_use_timing_independent` | Run the motivation pipeline under low and high `process_events()` cadence (or with/without explicit polling). Peak `torch.cuda.max_memory_allocated()` must be identical in both runs (modulo noise). Compare against `record_stream` variant where peak is polling-dependent. |
| 3 | `test_record_use_block_enters_pool_immediately` | `record_use` must not delay block return to pool. Use `memory_snapshot()` to assert block transitions to free state as soon as the ref drops — not via `cuda_events` queue. |
| 4 | `test_record_use_multiple_events` | Multiple `record_use` on one block → all waits issued at reuse. |
| 5 | `test_record_use_same_stream_is_noop` | `record_use(block->alloc_stream)` does not record or stash. |
| 6 | `test_record_use_coexists_with_record_stream` | Block with both attachments: record_stream's event-polling gates pool entry; record_use's waits are issued at subsequent alloc. Both mechanisms contribute correctly. |
| 7 | `test_record_use_split_block` | Block of size N with record_use; freed; next alloc takes N/2 (split). Returned half issues wait; remainder in pool has empty `pending_waits`; subsequent alloc of the other half does not wait. |
| 8 | `test_record_use_cross_stream_reuse` | Block alloc on A, `record_use(B)`, freed, reallocated by a request on *C*. C issues the wait. |

Test #2 is the design-critical test — it distinguishes the FIFO-chain mechanism from event polling.

### Step 9 — Update `_tensor_docs.py`

Rewrite the `record_use` docstring:

- Lead with the FSDP2 pattern as the intended semantics (FIFO chain, no polling).
- Explicit note that memory behavior is **timing-deterministic**, unlike `record_stream`.
- Example mirroring the design doc's motivation.
- Footgun note: call after last use.

### Step 10 — Dynamo / nested-tensor paths

No changes needed beyond what the current prototype has: the custom op + method-tracer in `_dynamo/variables/{streams,tensor}.py` and the jagged dispatch in `nested/_internal/ops.py` work regardless of the underlying allocator mechanism.

## Verification checklist

- [ ] `spin lint` clean.
- [ ] All 8 new tests pass.
- [ ] Existing `record_stream` tests pass (no regression).
- [ ] FSDP2 HSDP training test (`test_fully_shard_training.py` HSDP + bf16/fp32-reduce) passes — confirms no interference with the manual FIFO-chain pattern FSDP2 already uses.
- [ ] Memory-snapshot viz for a `record_use`-using workload: blocks transition free → pool → reused in one step; no entries in `cuda_events` attributable to `record_use`.

## Risk / edge-case checkpoints

1. **Allocation hand-off site.** Must run once per allocation, after block selection, with the requesting stream in scope, and under the allocator mutex. `alloc_found_block` fits all three.
2. **Block split ownership.** The `pending_waits` list is move-only (contains unique_ptrs). Use `std::move` into the returned child; the remainder gets a default-constructed empty vector. Verify via Test #7.
3. **Cross-device.** Skip + warn in PR 1 (Step 6).
4. **Empty block / null data_ptr.** `recordUse` is a no-op on null ptr (mirrors `recordStream`).
5. **Performance.** Empty-case `issue_pending_waits` is a single vector-empty check + branch — hot-path negligible. Non-empty case adds O(k) `cudaStreamWaitEvent` calls where k = number of `record_use` calls (typically 1).
6. **Coexistence with `record_stream` on the same block.** Free-path (`record_stream`): `insert_events` → `cuda_events` → `process_events` → `free_block` → pool. Alloc-path (`record_use`): `pending_waits` → `issue_pending_waits`. Orthogonal; both mechanisms gate reuse independently.

## Rollout

1. **This PR**: rework #181189 to the FIFO-chain mechanism above. All 8 new tests green.
2. **Follow-up**: FSDP2 migration — collapse `StreamHandoff` and the three NamedTuple helpers to `tensor.record_use(stream)` + plain `del`. Expected diff: -100 to -200 LOC in `_fsdp_param_group.py` / `_fsdp_collectives.py`.
3. **Follow-up**: `CUDAMallocAsyncAllocator` native implementation.
4. **Follow-up**: `c10/xpu` mirror.
5. **Follow-up**: lift same-device restriction.
6. **Separate RFC**: graph-capture-precise semantics.
