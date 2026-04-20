# Memory Visualization (torch/utils/viz)

This code visualizes CUDA memory snapshots produced by `torch.cuda.memory._snapshot()`. The visualization simulates the behavior of the CUDA caching allocator (`c10/cuda/CUDACachingAllocator.cpp`) by replaying trace events to reconstruct memory state over time — tracking allocations, frees, segment growth, pool isolation, and fragmentation.

## Architecture

- `process_alloc_data.js` — pure data processing, no DOM/d3 dependencies. Testable in Node.
- `MemoryViz.js` — d3-based rendering, imports from process_alloc_data.js via ESM.
- `test/profiler/test_memory_viz.js` — Node.js tests. Strips the ESM export line and eval's the module. Run with `node test/profiler/test_memory_viz.js`.

## Key concepts

### Two data sources in the pickle

The snapshot has `device_traces` (ring buffer of events) and `segments` (point-in-time state at `_snapshot()` time). These are independent: the trace may be truncated (ring buffer overflow) but segments are always complete. The visualization reconciles them — e.g., blocks in segments but not in the trace become "ghost blocks" or "initially_allocated".

### Memory isolation in the allocator

The CUDA caching allocator isolates free blocks by **(pool_id, stream)**. `get_free_block()` in `CUDACachingAllocator.cpp` searches within a single `BlockPool` and rejects blocks on different streams. This means freed memory in one (pool, stream) bucket cannot satisfy allocations in another, even if there's plenty of free space. This is why reserved memory can far exceed active memory.

Private pools (pool_id != (0,0)) additionally never unmap physical pages — `segment_unmap` events only occur for the default pool.

This isolation is only visualized in the "Allocated Memory (incl. Private Pools)" tab, which renders pool envelopes. The "Active Memory Timeline" tab ignores pool boundaries.
