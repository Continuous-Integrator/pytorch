# Design: Closing CPython Protocol Gaps with `generic_*` Functions

**Status**: Active — incremental implementation in progress
**Author**: anijain (with Claude)
**Date**: 2026-02-23 (revised 2026-03-16)

## Problem Statement

Dynamo's VariableTracker (VT) hierarchy re-implements CPython protocol methods
(`var_getattr`, `call_method`, hashing, equality, etc.) independently on each
VT subclass. This causes:

1. **CPython behavioral gaps**: Each VT subclass has its own approximation of
   CPython semantics. Fixing a gap in one VT doesn't fix it in others. Users
   hit these as unexpected graph breaks or silent incorrectness (e.g.,
   `functools.partial` as a dict key, truthiness of user objects).

2. **Code duplication**: The same CPython algorithm (e.g., the attribute lookup
   in `PyObject_GenericGetAttr`) is reimplemented across multiple VT classes
   with overlapping but divergent logic.

3. **Parallel protocol systems**: Some CPython behaviors have two independent
   code paths in Dynamo. For example, equality has both `is_python_equal` (a
   VT method, 26 implementations) and `call_method("__eq__")` (11+ VT
   implementations). Hashing has `is_python_hashable`/`get_python_hash` (32
   implementations) with zero `call_method("__hash__")` implementations.
   These parallel systems diverge and create gaps.

## Approach: Extract Shared `generic_*` Functions

Instead of a large-scale refactor (TypeVariableTracker with slot tables), we
take an incremental approach: extract the CPython algorithm for each protocol
into a shared free function with hooks on the base `VariableTracker`. Each VT
overrides only the hooks it needs.

This delivers most of the value (shared correct algorithms, closed CPython
gaps) with minimal churn. The `generic_*` functions can later become slot
implementations if we ever pursue a TypeVariableTracker design.

### Pattern

```python
# Shared algorithm in user_defined.py (or a new protocols.py)
def generic_<protocol>(tx, instance_vt, ...):
    """Mirrors CPython's PyObject_Generic<Protocol>."""
    # CPython algorithm with hook calls
    result = instance_vt.<hook_method>(tx, ...)
    ...

# Default hook on base VariableTracker (base.py)
class VariableTracker:
    def <hook_method>(self, tx, ...):
        """Default — correct for most VTs."""
        ...

# Override on specific VTs that need custom behavior
class UserDefinedObjectVariable(VariableTracker):
    def <hook_method>(self, tx, ...):
        """Custom — traces into descriptors, etc."""
        ...
```

## Protocol Inventory

### Completed

#### `generic_getattr` — `PyObject_GenericGetAttr` (tp_getattro)

**CPython algorithm**: MRO walk → data descriptor → instance dict → non-data
descriptor → plain class attr → `__getattr__` fallback → AttributeError.

**Implementation**: `user_defined.py:generic_getattr`. Shared across UDOV and
TensorVariable via hooks:

| Hook | Base default | UDOV override |
|---|---|---|
| `maybe_trace_getattribute` | Returns None (no override) | Checks for custom `__getattribute__` |
| `resolve_data_descriptor` | Calls `type.__getattribute__` | Traces into property fget, `__get__` |
| `resolve_type_attr` | Calls `type.__getattribute__` | Handles staticmethod, classmethod, etc. |
| `handle_getattr_fallback` | Calls the `__getattr__` function | Same + source tracking |
| `maybe_wrap_nn_module_source_for_instance` | Identity | NNModule source wrapping |

#### `generic_setattr` — `PyObject_GenericSetAttr` (tp_setattro)

**CPython algorithm**: MRO walk → type attr with `__set__` → instance dict
write. Note: only requires `__set__` on the type attr, NOT full data
descriptor status (`__get__` + `__set__`). This differs from getattr.

**Implementation**: `user_defined.py:generic_setattr`. Shared across UDOV and
UserFunctionVariable via hooks:

| Hook | Base default | UDOV override |
|---|---|---|
| `resolve_setattr_descriptor` | Writes to instance dict (ignores descriptor) | Traces into `__set__`, property fset |

### High Priority — Many VTs with Overlapping Incomplete Implementations

#### `generic_richcompare` — `tp_richcompare` (`__eq__`, `__ne__`, `__lt__`, etc.)

**CPython algorithm**:
```
1. If type(rhs) is a proper subclass of type(lhs) and overrides the op:
   try rhs.__rop__ first
2. Try lhs.__op__(rhs)
3. If result is NotImplemented, try rhs.__rop__(lhs)
4. If still NotImplemented: for __eq__ return identity check, for others TypeError
```

**Current Dynamo state**: Two parallel systems.
- `is_python_equal` / `is_python_equal`: 26 VT implementations. Used
  internally for dict keys, set membership, constant folding. Returns
  `bool` directly — no `NotImplemented` support, no reflected ops.
- `call_method("__eq__")`: 11+ VT implementations in `call_method`.
  Each handles it differently. Used when user code calls `==`.

VTs implementing `__eq__` in `call_method`:
BaseListVariable, RangeVariable, ConstDictVariable, DefaultDictVariable,
SetVariable, DictItemsVariable, TracebackVariable, TypingVariable,
TensorVariable, UserDefinedClassVariable, UserDefinedTupleVariable.

**Gaps**:
- Subclass priority (step 1) is not implemented in most VTs.
- `NotImplemented` return is rarely handled correctly.
- Reflected ops (`__req__`) are mostly missing.
- The two systems can give different answers for the same comparison.

**Plan**: Extract `generic_richcompare(tx, lhs, rhs, op)` that implements
the full CPython algorithm. Unify `is_python_equal` to delegate to this
(or at least share the same hook). Each VT provides a hook like
`richcompare_impl(tx, other, op)` that returns a result or
`NotImplemented`.

**Base class convention**: The base `VariableTracker.richcompare_impl`
(and the base hook for each future `generic_*` function) should raise
`unimplemented()` rather than returning a silent fallback. This ensures
that when a new VT is added and someone tries to compare it, the missing
implementation surfaces immediately as a visible graph break rather than
silently returning a wrong answer. Each concrete VT must explicitly
declare what it supports.

#### `generic_hash` — `tp_hash`

**CPython algorithm**:
```
1. Look up __hash__ on type(obj) via MRO
2. If __hash__ is None (explicitly unhashable): TypeError
3. Call __hash__(obj), must return int
4. If -1, replace with -2 (CPython implementation detail, skip)
```

**Current Dynamo state**: `is_python_hashable` (32 implementations) and
`get_python_hash` (26 implementations) are VT-level methods. No VT handles
`__hash__` via `call_method`. These are only used internally (dict keys,
set membership), never when user code calls `hash()`.

**Gaps**:
- New VTs often forget to implement `is_python_hashable` / `get_python_hash`.
- `hash()` in user code goes through `BuiltinVariable` isinstance cascades,
  not through the VT protocol.
- User-defined `__hash__` on UDOV instances isn't traced — it falls through
  to using `id()` or the real object's hash.

**Plan**: Extract `generic_hash(tx, instance_vt)` that walks the MRO for
`__hash__`, handles the `None` case (unhashable), and calls the method.
Unify `is_python_hashable` / `get_python_hash` to delegate to this.

#### `generic_bool` — `nb_bool` (truthiness)

**CPython algorithm**:
```
1. If type has __bool__: return bool(__bool__(obj))
2. If type has __len__: return len(obj) != 0
3. Return True (all objects are truthy by default)
```

**Current Dynamo state**: Almost nothing handles `__bool__` in `call_method`
(only TorchScriptObjectVariable). `BuiltinVariable.call_bool` has an isinstance
cascade. `InstructionTranslator` has ad-hoc truthiness checks in
`JUMP_IF_TRUE`/`JUMP_IF_FALSE` handlers.

**Gaps**:
- User-defined `__bool__` is rarely traced.
- The `__bool__` → `__len__` → `True` fallback chain is not implemented
  generically.
- ConstantVariable, ListVariable, DictVariable, etc. each handle truthiness
  differently.

**Plan**: Extract `generic_bool(tx, instance_vt)` implementing the 3-step
chain. Wire into `BuiltinVariable.call_bool` and the jump instruction
handlers.

### Medium Priority — Fewer VTs, But Correctness Matters

#### `generic_iter` — `tp_iter`

**CPython algorithm**:
```
1. If type has __iter__: return __iter__(obj)
2. If type has __getitem__: return a sequence_iterator(obj)
3. TypeError: object is not iterable
```

**Current Dynamo state**: 12+ VTs handle `__iter__` in `call_method`, each
returning its own iterator type. The `__getitem__` fallback (step 2) is not
implemented.

VTs with `__iter__` handling:
BaseListVariable, RangeVariable, RangeIteratorVariable, IteratorVariable,
ConstantVariable (str), ConstDictVariable, DictViewVariable,
DictItemsVariable, LocalGeneratorObjectVariable, NNModuleVariable,
NumpyNdarrayVariable.

**Gaps**:
- `__getitem__`-based iteration (step 2) is missing for user-defined objects.
- No unified "is iterable" check.

**Plan**: Extract `generic_iter(tx, instance_vt)` with the fallback chain.
Each VT provides a `tp_iter_impl` hook. Medium value — implementations are
type-specific (list returns list_iterator, dict returns dict_keyiterator),
so the shared function is mostly dispatch.

#### `generic_contains` — `sq_contains`

**CPython algorithm**:
```
1. If type has __contains__: return __contains__(obj, value)
2. Iterate obj, compare each element with value
```

**Current Dynamo state**: 8+ VTs handle `__contains__` in `call_method`.
Most delegate to `iter_contains` for the fallback.

VTs: BaseListVariable, RangeVariable, ConstantVariable, ConstDictVariable,
DictKeysVariable, GetAttrVariable, NNModuleVariable,
UserDefinedEnumClassVariable.

**Gaps**:
- The iterate-and-compare fallback exists (`iter_contains`) but isn't
  wired up generically.

**Plan**: Extract `generic_contains(tx, instance_vt, value_vt)`. Moderate
value since most VTs already delegate to `iter_contains`.

#### `generic_len` — `sq_length` / `mp_length`

**Current Dynamo state**: 7+ VTs handle `__len__`. Relatively simple — each
returns a constant. Low code sharing opportunity.

#### `generic_getitem` / `generic_setitem` — `mp_subscript` / `mp_ass_subscript`

**Current Dynamo state**: Many VTs handle `__getitem__`/`__setitem__`.
Implementations are heavily type-specific (list indexing vs dict lookup vs
tensor indexing). Less clear sharing opportunity.

### Lower Priority

- **`generic_call`** — `tp_call`: Mostly type-specific, less sharing.
- **`generic_repr`** / **`generic_str`**: Rarely needed during tracing.
- **`generic_next`** — `tp_iternext`: Type-specific iterator advancement.

## Recommended Execution Order

1. **`generic_richcompare`** + **`generic_hash`** — highest impact. Directly
   addresses the "functools.partial as dict key" class of CPython gaps. Unifies
   the parallel `is_python_equal`/`get_python_hash` and `call_method` systems.

2. **`generic_bool`** — quick win. Simple 3-step algorithm, currently missing
   entirely as a generic implementation.

3. **`generic_iter`** — addresses the `__getitem__`-based iteration gap.

4. **`generic_contains`** — small incremental improvement.

5. Remaining protocols as needed.

## Relationship to TypeVariableTracker

The original version of this design proposed a `TypeVariableTracker` class
mirroring CPython's `PyTypeObject` — a separate object carrying slot functions
(`tp_getattro`, `tp_setattro`, `tp_call`, etc.) that VT instances would point
to via `ob_type_vt`.

After analysis, we determined that `TypeVariableTracker` is functionally
equivalent to the `generic_*` + hooks approach for all protocols except the
**subclass cliff** (where `class MyList(list)` falls into UDOV and loses
ListVariable optimizations). The subclass cliff requires true type/instance
separation to solve — but it's not the primary user pain point. Users primarily
complain about CPython protocol gaps, not subclass handling.

The `generic_*` extraction approach:
- Delivers the same shared-algorithm benefits
- Requires no new abstraction layer
- Is incrementally adoptable (one protocol at a time)
- Produces functions that become natural slot implementations if we ever
  pursue TypeVariableTracker

If the subclass cliff becomes a top priority, the `generic_*` functions we're
building now become the slot implementations for `TypeVariableTracker`, so no
work is wasted.

## Current State of the Codebase

### VT protocol method counts (as of 2026-03-16)

| Protocol | VTs implementing | Mechanism |
|---|---|---|
| `var_getattr` | 45 overrides | VT method override |
| `call_function` | 62 overrides | VT method override |
| `is_python_hashable` | 32 implementations | VT method |
| `get_python_hash` | 26 implementations | VT method |
| `is_python_equal` | 26 implementations | VT method |
| `__eq__` in call_method | 11+ VTs | String dispatch |
| `__iter__` in call_method | 12+ VTs | String dispatch |
| `__contains__` in call_method | 8+ VTs | String dispatch |
| `__len__` in call_method | 7+ VTs | String dispatch |
| `__bool__` in call_method | 1 VT | String dispatch |
| `__hash__` in call_method | 0 VTs | Not implemented |
