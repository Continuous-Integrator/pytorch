# `Tensor.record_use`: FSDP2-style precise cross-stream lifetime in the caching allocator

Prototype: https://github.com/pytorch/pytorch/pull/181189 (note: the current prototype implements an earlier event-polling variant; the design here switches to the FIFO-chain mechanism described below — the PR needs to be updated to match).

## Purpose

Move the FSDP2 cross-stream lifetime pattern into the caching allocator as a one-call primitive, with identical timing semantics — **no `cudaEventQuery` polling, no dependence on allocator tick frequency**. The allocator reuses blocks by issuing `cudaStreamWaitEvent` on the requesting stream at allocation time, which is exactly what FSDP2 does today by hand.

## Motivation

The CUDA caching allocator tracks only each block's allocation stream; cross-stream reads are invisible to it. FSDP2 bridges this with a FIFO-chain pattern, repeated across three subsystems (all-gather output, reduce-scatter input, all-reduce output), ~40–80 LOC each:

```python
# Iteration N, on the comm/consumer stream:
with torch.cuda.stream(comm_stream):
    y = consumer_kernel(buf)
consumer_event = comm_stream.record_event()   # (1) "consumer done on comm"

# Hold buf alive across iterations so the allocator doesn't reclaim the
# block before the next iteration's wait.
comm_ctx.state = (buf, consumer_event)

# Iteration N+1: at the top of the next iter's comm work, gate on the
# prev consumer event before launching new work on the comm stream.
if prev := comm_ctx.state:
    comm_stream.wait_event(prev.event)         # (2) FIFO absorbs prev event
    comm_ctx.state = None                      # drops prev.buf — safe now,
                                               # because comm_stream's FIFO
                                               # orders reuse after the wait
# ...new allocation on comm_stream, new kernel, etc...
```

The load-bearing step is **(2)**: `comm_stream.wait_event(prev.event)` absorbs the consumer event into the allocation stream's FIFO *before* the next allocation that might reuse `prev.buf`'s block. Reuse is ordered by the stream's FIFO — deterministic in time, **no `cudaEventQuery` involved**. The stash in `comm_ctx.state` is there to keep the Python ref alive across the iteration boundary so that step (2) can read `prev.event` before dropping the tensor.

Every step is load-bearing in a different way:

- `consumer_event` must be recorded immediately after the consumer's last read, not later. Delay it and the event captures too much, over-reserving memory.
- The stash lifetime must outlive step (2). Drop the ref before the next iteration's wait_event and the allocator reclaims the block before any stream has waited on `consumer_event` → silent corruption.
- Step (2) must run on a stream whose FIFO ordering governs the reuse. For FSDP2 this is the comm stream (which is both the alloc stream and the next consumer).

FSDP2 PRs [#140044](https://github.com/pytorch/pytorch/pull/140044), [#179443](https://github.com/pytorch/pytorch/pull/179443), and [#180666](https://github.com/pytorch/pytorch/pull/180666) each traced to one of these going wrong in a new code path. The same recipe is reimplemented, with its own helper class, in activation-offloading hooks, non-FSDP collective libs, and user multi-stream code via `cpp_extension`. Each one is a potential UAF.

The critical observation: this pattern is **not** reducible to `Tensor.record_stream(stream)`. `record_stream` records its event at *block free time* and relies on the allocator's `cuda_events` / `process_events()` polling path to decide when to reuse — making memory footprint depend on `cudaEventQuery` timing. FSDP2 deliberately does **not** use `record_stream` for exactly this reason: production requires timing-deterministic memory behavior.

What's missing is an allocator-visible version of the FIFO-chain pattern — one that moves step (2) from user code into the allocator, without introducing polling.

## Proposal: `Tensor.record_use(stream)`

Same scene as above, rewritten with `record_use`:

```python
# Iteration N, on the comm stream:
with torch.cuda.stream(comm_stream):
    y = consumer_kernel(buf)
    buf.record_use(comm_stream)          # ← allocator stores this event on
                                          #   buf's block; will issue
                                          #   cudaStreamWaitEvent on the
                                          #   next stream that reuses the
                                          #   block, at allocation time
del buf                                  # any stream, any thread, any time
```

The stash, the cross-iteration `wait_event`, and the `with torch.cuda.stream(X):` drop wrapper all go away — the allocator holds the event directly on the block and issues the wait itself at the point of reuse.

### Precise semantics

1. **Event recorded at the call site.** `record_use(stream)` calls `cudaEventRecord` on `stream` now and attaches the event to the tensor's allocation block. The caller is expected to place the call right after their consumer's last read of the tensor on `stream`.
2. **Wait is issued at reuse time, on the reusing stream.** When the allocator hands this block (or a sub-range of it) out to a later allocation on some stream `R`, it first issues `cudaStreamWaitEvent(R, event)`. `R`'s FIFO now orders the new allocation after `event`. **No polling.**
3. **No-op on allocation stream.** If `stream == tensor.allocation_stream`, the call is a no-op — the alloc stream's FIFO already orders the consumer read and the next allocation.
4. **Accumulates.** Multiple `record_use` calls attach multiple events; all are issued as `cudaStreamWaitEvent` calls on the reusing stream at the point of reuse.
5. **Composes with `record_stream`.** A block may have both `record_use`-style waits and `record_stream`-style polled events; both gate reuse independently.
6. **Thread-safe.** Takes the same allocator mutex as `record_stream`.

The user-visible guarantee is the same as FSDP2's hand-rolled FIFO chain: once the tensor is dropped, the block is immediately available in the pool, and whoever picks it up will wait on the recorded event inside its own FIFO before using the memory. Peak memory is timing-independent.

## Comparison

| Aspect | `record_stream` | hand-rolled FSDP2 recipe | `record_use` (proposed) |
|---|---|---|---|
| Mechanism | event polling (`cudaEventQuery`) | FIFO chain (`cudaStreamWaitEvent` before alloc) | FIFO chain (`cudaStreamWaitEvent` at alloc) |
| Timing determinism | **depends on allocator poll cadence** | deterministic | deterministic |
| Event is recorded | at block free time | at caller's end-of-use | at caller's end-of-use |
| Caller code | 1 line | several load-bearing lines + helper class | 1 line |
| Safety if misused | always safe | UAF if any step wrong | UAF if called before last read |
| Python ref held by | caching allocator | user-owned Python stash | caching allocator |
| Drop-on-right-stream routing | `with stream(X): del` required | `with stream(X): del` required | automatic |
| Dynamo-traceable | yes | no (stream context + stash) | yes (custom op) |
| No-op on alloc stream | yes | caller must special-case | yes |
| Production-acceptable memory | no (timing-dependent) | yes | yes |
| BC | existing | n/a (user code) | additive; `record_stream` untouched |

The key row is **timing determinism**. `record_stream`'s "wait until `process_events()` polls and decides to release" is why FSDP2 avoids it. `record_use` uses exactly the same mechanism as FSDP2's manual pattern — stream wait events issued before the next kernel that might reuse the memory — so it inherits FSDP2's memory-footprint properties, not `record_stream`'s.

## Design

Scoped to the native CUDA caching allocator + a default fallback at the `DeviceAllocator` base so non-CUDA backends stay correct with zero code.

- New `Block::pending_waits` field: list of `(EventPool::Event, cuda::CUDAStream)` pairs representing events that any reusing stream must `cudaStreamWaitEvent` on before using this block's memory. Populated by `recordUse()`, cleared on first reuse.
- New `DeviceCachingAllocator::recordUse(block, stream)`: records `cudaEventRecord` eagerly on `stream`, appends the event to `block->pending_waits`. Same-stream calls are no-ops.
- **Allocation path**: after a block is selected from the pool and before it is returned, issue `cudaStreamWaitEvent(requesting_stream, event)` for each event in `block->pending_waits`, then clear the list. After this point the physical memory is ordered-after those events on the requesting stream's FIFO; any later reuse is safe by transitivity.
- **Free path is unchanged.** Unlike the original event-polling proposal, `pending_waits` does not gate when the block returns to the pool. Blocks return to the pool immediately on free (subject to the existing `stream_uses` path for `record_stream` callers), and the wait is paid at the next allocation.
- **No interaction with `cuda_events` / `process_events()`.** This proposal does not extend the polling path.
- `CUDAAllocator::recordUse` / `DeviceAllocator::recordUse` default to `recordStream`. `CUDAMallocAsyncAllocator`, `CUDAPluggableAllocator`, XPU stay correct with zero code (losing precision — acceptable as an opt-in improvement).
- ATen: one new `record_use` entry in `native_functions.yaml` + one-liner `RecordUse.cu`, mirroring `record_stream`.
- Python: `_tensor_docs.py` docstring + `overrides.py` stub + `_dynamo/variables/{streams,tensor}.py` custom op + method tracer + nested-tensor dispatch. All mirror `record_stream`'s wiring.

### Why wait at alloc-time, not free-time?

At free time, the allocator doesn't know which stream will pick up the block, so it cannot issue the wait. Storing the event and paying the wait at alloc time defers the decision to the exact moment a requesting stream is known. This is the same dynamic the FSDP2 recipe exploits manually — the wait_event is placed in the next iteration's code path, which runs on the stream that will do the reuse.

### Interaction with expandable segments / block splitting

On first reuse of any byte of the block, the wait is issued and `pending_waits` cleared. If the block is split, the sub-blocks share the (now-empty) `pending_waits` — once the physical memory is safe on one stream, it's safe period, because the event completing is a monotone condition. No per-sub-block bookkeeping needed.

## Risk and landability

1. **BC is clean.** `record_stream` semantics, `Block::stream_uses`, `insert_events()`, `process_events()`: all untouched. Every existing caller sees identical behavior.
2. **Timing determinism is the central property.** Memory is ordered by GPU stream dependencies, not by CPU-side polling. This is the production-required property that distinguishes this design from the earlier event-polling `record_use` variant.
3. **User-visible footgun.** `record_use` must be called after the consumer's last read or it's a UAF. Same hazard the hand-rolled recipe carries today; `record_use` inherits it, doesn't invent it.
4. **Graph-capture path.** `cudaEventRecord` inside capture participates in graph dependencies, which may or may not be desired. Simple option for PR 1: fall back to no-op (warn) under capture, mirror FSDP2's capture story. Worth a separate RFC if precise capture semantics are needed.
5. **Non-CUDA backends.** `DeviceAllocator::recordUse` default delegates to `recordStream`, so XPU / MallocAsync / pluggable allocators compile and run unchanged. Precise implementations are clean mirrors of the CUDA path; deferred to follow-up PRs.
6. **Allocator state footprint.** One extra vector per `Block`, empty on blocks that never record. No heap growth in steady state for `record_use`-free workloads.
7. **Cost at allocation.** Each `record_use` call issues one `cudaEventRecord`; each reuse of a tagged block issues one `cudaStreamWaitEvent`. Both are O(1) GPU-side and cheap. The allocator is already on the critical allocation path, so adding a conditional wait_event is negligible.

## Alternatives (why not…)

- **Event-polling `record_use` (the earlier revision of this doc).** Stashes events in `cuda_events`; reuse is gated by `process_events()` polling. Rejected because memory footprint becomes timing-dependent, which is the reason production code avoids `record_stream` in the first place.
- **Make `record_stream` precise.** BC-breaking; and still leaves the event-polling mechanism in place.
- **`keep_alive_until(event)` (caller supplies the event).** Forces boilerplate at the common call site; addable later as a companion.
- **`record_stream(stream, precise=True)`.** Behavior-changing flag on a public API is hard to grep and hard to review.
- **Storage-level API.** `record_stream` is a `Tensor` method; mirror that surface.
- **Name.** Placeholder. Candidates: `defer_reuse_until`, `queue_wait_on_reuse`, `record_use`, `stream_barrier`. Bikeshed before public-API commitment.

## Scope

**PR 1:** native CUDA allocator implementation of the FIFO-chain mechanism above, `Tensor.record_use`, docs, dynamo custom op, tests that (a) verify precision vs `record_stream`, (b) verify **timing independence** (run under artificially slow/fast polling or multi-stream contention and observe no change in peak memory).

**Follow-ups:** `CUDAMallocAsyncAllocator` native path; `c10/xpu` mirror; FSDP2 migration (collapse `StreamHandoff` to a one-liner, drop the subsystem-specific NamedTuples); capture-precise semantics (separate RFC).

**Non-goals:** replacing `record_stream`; changing `stream_uses` semantics; runtime misuse detection; any code path that uses `cudaEventQuery` polling.

## Open questions

1. **Name.** `record_use` vs alternatives above.
2. **FSDP2 migration.** Once this primitive lands, does FSDP2 want to adopt it (deleting `StreamHandoff` + NamedTuple helpers), or keep the explicit recipe for reviewability? Either is fine as long as the primitive exists for other users.
3. **Capture story.** Does precise-inside-capture need to exist, or is "no-op under capture" acceptable given that FSDP2 also doesn't use this pattern under capture?
4. **Consumer-stream coverage.** FSDP2's recipe uses a *single* event (the consumer_event on the comm stream). Is there a real use case for multiple attached events on one block, or should the API just take one event and simplify?

## References

- `Tensor.record_stream` docstring (`torch/_tensor_docs.py`).
- FSDP2 `_fsdp_param_group.py`, `_fsdp_collectives.py` — the source pattern.
- Jane Xu, "FSDP & CUDACachingAllocator: an outsider newb perspective" (dev-discuss.pytorch.org).
- `cudaMallocAsync` / `cuMemAsyncFree` — allocator-aware-of-streams at driver level; doesn't compose with native caching allocator.

## Appendix: prototype status

The PR linked at the top implements an earlier event-polling variant of `record_use`, which is **superseded by this design**. A working prototype of the FIFO-chain mechanism above is not yet in the PR; the existing allocator changes need to be reworked to use `cudaStreamWaitEvent` at alloc-time rather than attaching to `cuda_events`. This is a smaller diff than the polling variant (no `consume_use_events`, no changes to `free()` or `process_events()`) and will replace the current prototype.
