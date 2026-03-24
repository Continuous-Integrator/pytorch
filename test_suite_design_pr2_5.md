# Minimal Test Suite Design: PR2–5 symm_mem CUDAGraph Stack

## Principle

Each PR carries **only** the minimal tests that cover its delta.
Tests from lower PRs are not modified unless the implementation they verify has changed.

---

## Stack Overview

| PR | Hash | What it does |
|----|------|-------------|
| PR1 | landed | `ExternKernelOut` lowering, `register_out_variant()` |
| PR2 | fe15df24 | symm_mem planning: auto-copy placeholder→P2P, CommBufferLayout upstream propagation |
| PR3 | b200dfe3 | CUDAGraph P2P pool handling: `p2p_input_idxs`, `_check_liveness` skip |
| PR4 | 185044a6 | Hoist `ExternKernelOut` allocs into prior CG partition |
| PR5 | 9160796e | Layout allocator approach: `AllocatorType`, persistent P2P buf, DMA `.copy_()` |

---

## Current State — Test × PR Matrix

| Test | PR2 | PR3 | PR4 | PR5 |
|------|-----|-----|-----|-----|
| `test_external_allocation_fallback` | **add** (codegen) | **modify** → CG replay | — | — |
| `test_symm_mem_placeholder_auto_copy` | **add** (codegen) | — | — | **modify** → Layout approach, add Path3 |
| `test_symm_mem_upstream_propagation` | **add** (codegen) | — | — | — |
| `test_cudagraph_p2p_input_passthrough` | — | **add** | — | — |
| `test_symm_mem_upstream_propagation_cudagraph` | — | **add** | — | — |
| `test_hoisting_with_device_copy` | — | — | **add** | — |
| `test_layout_allocator_type_propagation` | — | — | — | **add** |

### Problems Identified

1. **PR2: `test_external_allocation_fallback` ≈ `test_symm_mem_placeholder_auto_copy`**
   Both test the identical scenario (regular CUDA placeholder → `one_shot_all_reduce`).
   Only diff: `placeholder_auto_copy` additionally asserts `one_shot_all_reduce_out`.
   **Redundant in PR2.**

2. **PR3 modifies PR2's `test_external_allocation_fallback`**
   Converts it from a codegen test to a CG replay test (removes `assertIn("empty_strided_p2p")`,
   adds `mode="reduce-overhead"` + multi-iteration loop). This is a layering violation —
   PR3 should add its own CG test, not rewrite PR2's test. Side effect: after PR3, the
   PR2 redundancy *accidentally* becomes a codegen/CG complementary pair, but the test name
   `test_external_allocation_fallback` no longer describes what it actually tests.

3. **PR5 modifies PR2's `test_symm_mem_placeholder_auto_copy`**
   Justified (implementation changed → assertions must follow), but entangled with problem #1:
   PR5 is modifying a test that shouldn't exist separately in the first place.

---

## Proposed Design

### PR2 — Two codegen tests

| Test | Scenario | Key assertions | Covers |
|------|----------|---------------|--------|
| `test_symm_mem_placeholder_auto_copy` | placeholder → allreduce | `empty_strided_p2p`, `one_shot_all_reduce_out`, correctness | Auto-copy insertion for graph inputs |
| `test_symm_mem_upstream_propagation` | mm → cpu → cuda → add → allreduce | `empty_strided_p2p`, `one_shot_all_reduce_out`, correctness | CommBufferLayout upstream propagation through pointwise across fallback boundary |

**Change:** Delete `test_external_allocation_fallback`. Merge its (only unique) assertion
into `test_symm_mem_placeholder_auto_copy`. Or equivalently: keep `test_external_allocation_fallback`,
add the missing `one_shot_all_reduce_out` assertion, delete `test_symm_mem_placeholder_auto_copy`.
Pick one name, kill the other.

### PR3 — Three new CG tests, no modifications to PR2

| Test | Scenario | Key assertions | Covers |
|------|----------|---------------|--------|
| `test_placeholder_auto_copy_cudagraph` (NEW name) | placeholder → allreduce, `mode="reduce-overhead"`, 3 iterations | CG replay correctness | `p2p_input_idxs` for graph placeholder inputs |
| `test_cudagraph_p2p_input_passthrough` | graph_a(mm) → graph_b(allreduce), 2 compiled graphs, 4 iterations | CG replay correctness | P2P tensor crosses CG graph boundary without re-alloc |
| `test_symm_mem_upstream_propagation_cudagraph` | mm → cpu → cuda → add → allreduce, `graph_partition=True, triton.cudagraphs=True`, 3 iterations | CG replay correctness | CG replay of upstream propagation pattern |

**Change:** Stop modifying `test_external_allocation_fallback`. Create
`test_placeholder_auto_copy_cudagraph` as a standalone new test — same scenario as
PR2's placeholder test but with CG enabled.

### PR4 — One new test, independent

| Test | Scenario | Key assertions | Covers |
|------|----------|---------------|--------|
| `test_hoisting_with_device_copy` | x+1 → cpu → cuda → ×2, no symm_mem | partition_0 has ≥2 `empty_strided_cuda`, CG replay correctness | Hoisted ExternKernelOut alloc in prior CG partition |

**No change needed.** Clean, independent, tests the general hoisting mechanism without
symm_mem coupling.

### PR5 — Modify PR2's test + one new unit test

| Test | Scenario | Key assertions | Covers |
|------|----------|---------------|--------|
| `test_symm_mem_placeholder_auto_copy` (MODIFY) | Same as PR2, but assertions updated | `empty_strided_p2p`, `.copy_(`, `one_shot_all_reduce_out`, `_p2p_buf_` at module level; **append** Path3 fallback section (cpu→cuda→allreduce, no `_p2p_buf_`) | Layout approach replaces Triton identity copy; fallback path still works |
| `test_layout_allocator_type_propagation` (NEW) | Unit test, no distributed | `as_fixed()` propagates allocator, `__eq__` distinguishes allocators | AllocatorType plumbing in ir.py |

**PR5's modification of PR2's test is justified** — the implementation changed
(Triton identity copy → DMA `.copy_()`), so the codegen assertions must update.
This is the one legitimate case of modifying a lower PR's test.

---

## Clean Dependency Graph

```
PR2 tests (codegen layer):
  ├── test_symm_mem_placeholder_auto_copy     ← placeholder → P2P auto-copy
  └── test_symm_mem_upstream_propagation      ← CommBufferLayout propagation

PR3 tests (CG layer, all new, no PR2 modifications):
  ├── test_placeholder_auto_copy_cudagraph    ← CG variant of PR2 placeholder test
  ├── test_cudagraph_p2p_input_passthrough    ← cross-graph P2P passing
  └── test_symm_mem_upstream_propagation_cudagraph  ← CG variant of PR2 upstream test

PR4 test (general, independent):
  └── test_hoisting_with_device_copy          ← no symm_mem, tests hoisting

PR5 tests (Layout allocator layer):
  ├── test_symm_mem_placeholder_auto_copy     ← MODIFY: update assertions for Layout approach
  └── test_layout_allocator_type_propagation  ← NEW: unit test for AllocatorType
```

## Action Items

| # | PR | Action |
|---|-----|--------|
| 1 | PR2 | Delete `test_external_allocation_fallback`. Add `one_shot_all_reduce_out` assertion to `test_symm_mem_placeholder_auto_copy`. |
| 2 | PR3 | Stop modifying `test_external_allocation_fallback`. Add new `test_placeholder_auto_copy_cudagraph` instead (same scenario, CG mode). |
| 3 | PR3 | Keep `test_cudagraph_p2p_input_passthrough` and `test_symm_mem_upstream_propagation_cudagraph` as-is. |
| 4 | PR4 | No changes. `test_hoisting_with_device_copy` is clean. |
| 5 | PR5 | Modify `test_symm_mem_placeholder_auto_copy` (now the sole placeholder test from PR2) for Layout approach. Keep `test_layout_allocator_type_propagation`. |
