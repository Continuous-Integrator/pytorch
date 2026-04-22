# `Tensor.record_use`: precise cross-stream lifetime for the CUDA caching allocator

Draft design doc — April 2026

Audience: PyTorch core / runtime / caching-allocator maintainers.

## Motivation

The CUDA caching allocator tracks only each block's allocation stream;
cross-stream reads are invisible to it. FSDP2 bridges this with a
five-step recipe, repeated in three subsystems (all-gather output,
reduce-scatter input, all-reduce output) at ~40–80 LOC each:

```python
# Producer on stream A (compute), consumer on stream B (comm).
stream_B.wait_event(producer_event)
with torch.cuda.stream(stream_B):
    y = consumer_kernel(x)                 # e.g. reduce_scatter
consumer_event = stream_B.record_event()

# Keep a Python ref alive past this scope — the allocator would
# otherwise reclaim x's block before consumer_event fires.
stash[slot] = x

# ...later, after some path has waited for consumer_event...
with torch.cuda.stream(stream_B):          # critical: drop inside the
    stash[slot] = None                     # consumer's stream context
```

Three invariants make it work:

- `consumer_event` is recorded before any unrelated work queues on
  `stream_B`. Delay it and the event captures too much, over-reserving
  memory.
- `stash[slot]` lifetime matches `consumer_event` pending. Shorter →
  use-after-free; longer → O(N_layers) memory leak.
- The drop runs inside `with torch.cuda.stream(stream_B):`. The
  allocator attributes each free to the current stream at the moment
  of deletion, and only `stream_B`'s FIFO has absorbed `consumer_event`;
  drop without the wrapper and a later allocation reuses the block
  while the consumer is still reading.

FSDP2 PRs [#140044](https://github.com/pytorch/pytorch/pull/140044),
[#179443](https://github.com/pytorch/pytorch/pull/179443), and
[#180666](https://github.com/pytorch/pytorch/pull/180666) each traced
to one of these going wrong in a new code path. The same recipe, with
its own helper class, is reimplemented in activation-offloading hooks,
non-FSDP collective libs, and user multi-stream code via
`cpp_extension`.

The caching allocator already has the event-polling machinery
(`cuda_events`, `event_count`, `process_events()`). The only
user-facing entry into it today, `Tensor.record_stream(stream)`,
records at block-free time rather than at consumer-done — so
production code doesn't use it and hand-rolls the recipe instead.

## Proposal: `Tensor.record_use(stream)`

The same scene from the Motivation section, rewritten with `record_use`:

```python
stream_B.wait_event(producer_event)
with torch.cuda.stream(stream_B):
    y = consumer_kernel(x)
    x.record_use(stream_B)           # ← allocator-visible barrier
del x                                # any stream, any thread, any time
```

The `consumer_event = stream_B.record_event()`, the `stash[slot] = x`,
the deferred `with torch.cuda.stream(stream_B): stash[slot] = None`,
and the subsystem-specific NamedTuple that glued them together all go
away. Drop order, drop site, and drop stream stop mattering.

`record_use` records a fresh CUDA event on `stream` at the point of the
call and attaches it to the tensor's allocation block; the caching
allocator will not reuse the block until every attached event has
fired. Precise semantics:

1. **Event recorded now.** The allocator calls `cudaEventRecord` on
   `stream` at the call site. The caller is expected to place the call
   right after their consumer's last read of the tensor on `stream`.
2. **No-op on allocation stream.** `record_use(tensor.alloc_stream)`
   is a no-op — the allocation stream's FIFO already orders the read
   and the next allocation.
3. **Accumulates.** Multiple `record_use` calls attach multiple events.
   The block waits for all of them.
4. **Composes with `record_stream`.** A block may have both precise
   `use_events` and imprecise `stream_uses`. Both gate the free.
5. **Thread-safe.** Takes the same allocator mutex as `record_stream`.

## Design

Scoped to the native CUDA caching allocator + a default fallback at
the `DeviceAllocator` base so non-CUDA backends stay correct with zero
code. The shape:

- New `Block::use_events` field: list of `(EventPool::Event, CUDAStream)`
  pairs. Events come from the allocator's existing `EventPool` — no new
  resource.
- New `DeviceCachingAllocator::recordUse(block, stream)`: records
  `cudaEventRecord` eagerly on `stream` and stashes on the block.
  Same-stream and under-capture calls degrade to the existing
  `stream_uses` path.
- Existing `free()` gate extends from `!stream_uses.empty()` to
  `!stream_uses.empty() || !use_events.empty()`; the new
  `consume_use_events(block)` feeds the pre-recorded events straight
  into the existing `cuda_events` polling queue. `process_events()`
  and the `event_count → 0` free discipline are unchanged.
- `CUDAAllocator::recordUse` / `DeviceAllocator::recordUse` default to
  `recordStream`. `CUDAMallocAsyncAllocator`, `CUDAPluggableAllocator`,
  XPU stay correct-but-imprecise without any code change.
- ATen: one new `record_use` entry in `native_functions.yaml` +
  one-liner `RecordUse.cu`, mirroring `record_stream` exactly.
- Python: `_tensor_docs.py` docstring + `overrides.py` stub +
  `_dynamo/variables/{streams,tensor}.py` custom op + method tracer
  (preserves `torch.compile` coverage from day one) + nested-tensor
  dispatch. All mirror `record_stream`'s wiring.

Full diffs, code, and edge-case handling live in the prototype PR.

## Comparison

| Aspect | `record_stream` | hand-rolled recipe (prod today) | `record_use` (proposed) |
|---|---|---|---|
| Event is recorded | at block free time | at caller's end-of-use | at caller's end-of-use |
| Precision | conservative | precise | precise |
| Caller code | 1 line | 5-ish load-bearing lines | 1 line |
| Safety if misused | always safe | UAF if any step wrong | UAF if called before last read |
| Ref held alive by | caching allocator | user-owned Python stash | caching allocator |
| Drop-on-right-stream routing | `with stream(X): del` required | `with stream(X): del` required | automatic |
| Dynamo-traceable | yes | no (stream context + stash) | yes (custom op) |
| No-op on alloc stream | yes | caller must special-case | yes |
| Composable | one call per stream | one stash per concurrent use | one call per use point |
| Graph capture | yes (via deferral) | works but requires care | falls back to `record_stream` for PR 1 |
| Allocator overhead | 1 `cudaEventRecord` per stream at free | 1 user-level event per use | 1 `cudaEventRecord` per call |
| BC | existing | n/a (user code) | additive; `record_stream` untouched |

## Risk and landability

Things TL review should weigh:

1. **BC is clean.** `record_stream` semantics, `Block::stream_uses`
   semantics, `insert_events()` behavior, `process_events()` loop:
   all untouched. Every existing caller sees identical behavior.
2. **User-visible footgun.** `record_use` must be called after the
   consumer's last read or it's a UAF. This is the same hazard
   today's hand-rolled callers already carry; `record_use` inherits
   it, doesn't invent it. Docstring is the guard (no runtime checker
   — infeasible without new instrumentation).
3. **Graph-capture path degrades silently to imprecise.** Under
   capture, `recordUse` falls back to `stream_uses`. Correct, loses
   precision. Acceptable for PR 1; a precise-in-capture story needs a
   separate RFC coordinated with `graph_capture_record_stream_reuse`.
4. **Move-only `Block`.** `use_events` contains `unique_ptr`s, so
   `Block` becomes move-only. Required one `std::move` in
   `AllocParams` assignment — the only Block copy in the file. No
   other spots use Block by value.
5. **Codegen plumbing.** `record_use` needs entries in three allow-lists
   (`FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT`,
   `MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION`, and the non-differentiable
   list in `gen_variable_type.py`). Mechanical, same pattern as
   `record_stream`.
6. **Non-CUDA backends.** `DeviceAllocator::recordUse` default
   delegates to `recordStream`, so XPU / MallocAsync / pluggable
   allocators compile and run unchanged. Precise implementations are
   clean mirrors of the CUDA path; deferred to follow-up PRs.
7. **Allocator state footprint.** One extra vector per `Block`, empty
   on blocks that never record. No heap growth in steady state for
   `record_use`-free workloads.
8. **Snapshot visibility gap.** `use_events` doesn't appear in
   `SnapshotInfo`. Open question below.

## Alternatives (why not…)

- **Make `record_stream` precise.** BC-breaking: any caller that today
  calls it earlier than their last read would silently become a UAF.
- **`keep_alive_until(event)` (caller supplies event).** Forces
  boilerplate at the common call site; could be added later as a
  companion.
- **`record_stream(stream, precise=True)`.** Behavior-changing flag
  on a public API is hard to grep and hard to review.
- **Storage-level API.** `record_stream` is a `Tensor` method; mirror
  that surface.
- **Name.** Placeholder. Candidates:
  `record_stream_precise`, `stream_barrier`, `keep_alive_until`,
  `done_using_on`. Bikeshed before public-API commitment.

## Scope

**PR 1:** native CUDA allocator, `Tensor.record_use`, docs, dynamo
custom op, 5 unit tests.

**Follow-ups:** `CUDAMallocAsyncAllocator` native path; `c10/xpu`
mirror; FSDP2 migration (collapses `StreamHandoff` to a one-liner);
capture-precise (separate RFC); `torch.compile` eager-mode fast path.

**Non-goals:** replacing `record_stream`; changing `stream_uses`
semantics; runtime misuse detection.

## Open questions

1. **Name.** `record_use` vs alternatives above.
2. **Pathological per-block event accumulation.** Tight-loop
   `record_use` without intervening frees grows the per-block vector.
   Worth a cap + `TORCH_WARN_ONCE`, or trust users?
3. **Snapshot visibility.** Should `use_events` surface in
   `SnapshotInfo` / memory-viz tooling? Free is already attributed
   correctly via `process_events`; it's the stash window that's
   currently invisible.
4. **Capture story.** Long term, should precise events participate in
   captured graphs' dependency structure, or always defer to
   post-capture? Concrete use cases would help decide.

## References

- `Tensor.record_stream` docstring (`torch/_tensor_docs.py`).
- Jane Xu, "FSDP & CUDACachingAllocator: an outsider newb perspective"
  (dev-discuss.pytorch.org).
- `cudaMallocAsync` / `cuMemAsyncFree` — allocator-aware-of-streams
  approach; driver-level, doesn't compose with the native caching
  allocator's pool semantics.

## Appendix: prototype

Implemented and passing locally. 5 new `record_use` tests green + 3
existing `record_stream` tests green (no regression). Build clean on
the native CUDA allocator path. File-by-file diff, code, and build
plumbing live in the PR.
