# Design Exploration: In-Place Buffer Reuse for Custom Ops

**Related:** [#164696](https://github.com/pytorch/pytorch/issues/164696), #175116 (out-variant, landed)

---

## Goal

Enable `torch.compile` to reuse a custom op's **dead input buffer as its output buffer**, skipping the output allocation entirely. The user provides a semantic annotation; the compiler decides when it's safe.

Analogous to how `torch.Tag.out_variant` cleanly enables out-variant discovery, we want a clean annotation + workflow for in-place buffer reuse.

---

## The Semantic Question

A custom op `y = op(x)` can reuse `x`'s memory for `y` **only if**:
1. The user guarantees the kernel's access pattern allows aliasing (e.g., elementwise)
2. The compiler verifies `x` has no other live users after this op

(1) is opaque to the compiler — it must come from the user.
(2) is already something the compiler knows how to check (used for Triton `inplace_buffers` today).

**Core design question: What is the right annotation, and where does the reuse decision happen?**

---

## Annotation Design Options

### Option A: Tag on the out-variant

Extend the existing out-variant mechanism. The user already has:
```python
# Functional
@custom_op("myns::op", mutates_args=())
def op(x: Tensor) -> Tensor: ...

# Out-variant
@custom_op("myns::op.out", mutates_args=("out",), tags=[torch.Tag.out_variant])
def op_out(x: Tensor, *, out: Tensor) -> None: ...
```

Add a new tag that says "out can alias x":
```python
tags=[torch.Tag.out_variant, torch.Tag.allow_input_output_aliasing]
```

**Pro:** Minimal new API surface; reuses existing out-variant infra.
**Con:** Doesn't specify *which* input can alias *which* output (matters for multi-input/output ops).

### Option B: Explicit aliasing map

```python
@custom_op("myns::op.out", mutates_args=("out",), tags=[torch.Tag.out_variant])
def op_out(x: Tensor, *, out: Tensor) -> None: ...

# Separate registration
register_aliasable_args(
    torch.ops.myns.op.out,
    aliasing_map={"out": "x"},  # out can alias x
)
```

**Pro:** Precise; handles multi-input/output (e.g., `{"out1": "a", "out2": "b"}`).
**Con:** More boilerplate; separate registration step.

### Option C: Schema-level annotation

Define aliasing constraints directly in the op schema, similar to how `mutates_args` works:
```python
@custom_op("myns::op.out", mutates_args=("out",), aliasable_pairs=[("out", "x")], tags=[torch.Tag.out_variant])
def op_out(x: Tensor, *, out: Tensor) -> None: ...
```

**Pro:** Self-contained; discoverable from the schema alone.
**Con:** New schema concept; needs upstream changes.

---

## Where Does Reuse Happen?

### Path 1: Lowering (FX pass / IR construction)

At graph construction time, detect the annotation and directly pass the input buffer as the `out=` argument instead of allocating a new one.

```python
# Normal out-variant lowering (current, from #175116):
buf1 = torch.empty(...)
op.out(x=buf0, out=buf1)

# With in-place reuse at lowering:
op.out(x=buf0, out=buf0)  # no allocation
```

**Challenge:** At lowering time, we don't yet know if `buf0` is dead after this op. Liveness is a scheduler concern. Lowering would have to be conservative or defer the decision.

### Path 2: Scheduler (`decide_inplace_update`)

Keep lowering as-is (allocate `buf1`). Then in the scheduler, detect that `buf0` and `buf1` are same-sized, `buf0` is dead, and the op has the aliasing annotation → rewrite `buf1 = buf0` (the existing `inplace_buffers` mechanism).

**This requires fixing two blockers:**
- `ExternKernelSchedulerNode.can_inplace()` → currently returns `False`
- `decide_inplace_update()` → currently skips `ExternKernelSchedulerNode`

**Pro:** Liveness info is available; matches how Triton `inplace_buffers` already works.
**Con:** Requires scheduler changes for extern kernels.

### Path 3: Hybrid — reinplace pass (FX-level, post-functionalization)

The `reinplace` pass already runs at the FX level and converts functional ops to mutating equivalents when inputs are dead. Could extend it to handle custom ops with the aliasing annotation.

**Pro:** Already has liveness analysis at FX level; runs before scheduling.
**Con:** `reinplace` operates on `auto_functionalized` patterns — may need generalization.

---

## Open Questions

1. **Annotation design:** Tag vs. explicit map vs. schema-level — which gives the cleanest UX with the least boilerplate?

2. **Lowering vs. scheduler vs. reinplace:** Where should the reuse decision live? Scheduler has the best liveness info but requires extending `ExternKernelSchedulerNode`. Reinplace is FX-level and already does similar analysis. Lowering is too early without liveness.

3. **Interaction with backward:** If `x` requires grad, autograd saves it — `x` won't be dead, so reuse won't trigger. Is that sufficient, or do we need an explicit guard?

4. **Multi-output ops:** For `(y1, y2) = op(a, b)` where `y1` can alias `a` and `y2` can alias `b` — do we need both inputs to be dead simultaneously, or can we partially alias?

5. **Validation:** Should there be a debug mode that runs the op both ways (aliased and non-aliased) and checks numerical equivalence?
