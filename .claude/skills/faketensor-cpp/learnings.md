# FakeTensor C++ — Debugging Learnings

Findings from debugging sessions, organized by category. Each entry includes the date, symptom, root cause, and fix.

## Common Device Logic

<!-- Add entries like:
### YYYY-MM-DD: [short title]
**Symptom:** ...
**Root cause:** ...
**Fix:** ...
-->

## Dispatch Key Issues

### 2026-03-23: CompositeExplicitAutograd ops crash with symbolic shapes
**Symptom:** `test_symbolic_shapes_split` crashes with `dispatch_on_subclass called with NO overloaded args that actually triggered dispatch` from `python_arg_parser.cpp:648`.
**Root cause:** `TensorImpl::sizes_custom()` (`c10/core/TensorImpl.cpp:344`) has a check for `has_symbolic_sizes_strides_` that routes to `pyobj_interpreter->sizes()` even when the tensor has no Python dispatch support. The call chain: CompositeExplicitAutograd kernel (e.g. `split`) calls `self.size(dim)` → `sizes_custom()` → sees `has_symbolic_sizes_strides_=true` → `pyobj_interpreter->sizes()` → `torchDispatchFromTensorImpl()` → tries `__torch_dispatch__` dispatch → fails because C++ fake tensors are plain `torch.Tensor` with no `__torch_dispatch__` and no `TorchDispatchMode` active.
**Why Python FakeTensor works:** Python FakeTensor IS a tensor subclass with `__torch_dispatch__`, and `FakeTensorMode` IS a `TorchDispatchMode`, so the dispatch finds a handler.
**Why only symbolic shapes:** Without symbolic sizes, `has_symbolic_sizes_strides_` is false, so `sizes_custom()` returns `sizes_default()` without going through Python.
**Why only some ops:** Ops with proper Meta kernels (e.g. `mm`, `cat`) use `sym_size()` internally, which goes through `sym_sizes_custom()` → `sym_sizes_default()` (pure C++, no Python needed). Only CompositeExplicitAutograd ops that call `self.size()` (the concrete `int64_t` path) trigger the crash. `split` is one such op (`aten/src/ATen/native/TensorShape.cpp:3174` and `TensorShape.h:61`).
**Key distinction:** `sizes_custom()` unconditionally routes symbolic tensors to Python. `sym_sizes_custom()` only routes to Python if `matches_python_custom()` is true (checking Python key), otherwise falls through to `sym_sizes_default()` which reads `symbolic_shape_meta().sizes_` in pure C++. Same pattern applies to `strides_custom()` vs `sym_strides_custom()`.
**Fix:** Not yet implemented. Options: (A) Push a Python `TorchDispatchMode` that handles sizes/strides, (B) Use decomposition tables to avoid CompositeExplicitAutograd kernels, (C) Register op-specific Fake kernels that use `sym_size()`.
**Affected code paths:** `sizes_custom()`, `strides_custom()`, `numel_custom()` — all have the same `has_symbolic_sizes_strides_` check that assumes Python dispatch support.

## TLS / Mode Lifecycle

### 2026-03-23: IncludeDispatchKeyGuard(Meta) leaks MetaBit into TLS
**Symptom:** After exiting C++ fake tensor mode, `torch.randn()` hits infinite recursion through `_refs.normal_` → `prims.normal` → `_normal_aten` → `normal_` → ... because MetaBit is stuck in TLS included, corrupting dispatch for all subsequent ops.
**Root cause:** `DispatchKeySet::operator-` (used by `IncludeDispatchKeyGuard` destructor) **only removes functionality keys, never backend bits** (see `DispatchKeySet.h` comment: "ONLY for the functionality keys. Any backend bits set on self will remain unchanged"). So `IncludeDispatchKeyGuard(DispatchKey::Meta)` adds both `Dense` (functionality) + `MetaBit` (backend) on construction, but the destructor only removes `Dense`, permanently leaking `MetaBit`.
**How Python avoids this:** `in_kernel_invocation_manager` uses `_PreserveDispatchKeyGuard()` which saves/restores the **entire** TLS keyset (both functionality and backend bits) via snapshot/restore, sidestepping the `operator-` limitation entirely. The comment at line 603-604 even acknowledges the Dense leaking issue.
**Fix:** Replace `IncludeDispatchKeyGuard` + `ExcludeDispatchKeyGuard` with `ForceDispatchKeyGuard` (C++ equivalent of `_PreserveDispatchKeyGuard`). Save the entire TLS keyset, then manually set included/excluded via `_force_tls_local_dispatch_key_set`. The guard's destructor restores the original keyset exactly.
**Also needed:** Meta must be added to the `dispatchKeySet` passed to `redispatchBoxed` (not just TLS), because `redispatchBoxed` uses the passed-in keyset directly without recomputing from TLS. In the original code Meta was always in TLS so the initial dispatch keyset already contained it; with lazy activation, it must be added explicitly.
**Rule:** Never use `IncludeDispatchKeyGuard` for per-backend dispatch keys (those with backend components like MetaBit, CUDABit, etc.). `operator-` won't clean up the backend bits. Use `ForceDispatchKeyGuard` instead.

### 2026-03-23: Meta should only be in TLS during kernel invocation, not mode lifetime
**Symptom/motivation:** Original C++ `FakeTensorModeTLS::set_state` eagerly added Meta to TLS included for the entire lifetime of the mode. This works today (Fake has higher priority), but diverges from Python's behavior where Meta is only added at kernel invocation time inside `in_kernel_invocation_manager`.
**Fix:** Removed Meta manipulation from `set_state`/`reset_state` (they now only toggle the Fake key). Meta is added locally in `fakeFallback` right before redispatch, scoped by `ForceDispatchKeyGuard` for automatic cleanup.

### 2026-03-23: Three keysets in the dispatcher — tensor, TLS, and dispatch keyset
**Context:** Understanding why both TLS modification AND dispatch keyset modification are needed when redispatching to the meta kernel from `fakeFallback`.
**Key facts:**
1. **Tensor keyset** (`tensor.key_set()`): lives on `TensorImpl`, describes what the tensor is. Immutable during dispatch.
2. **TLS keyset** (`tls_local_dispatch_key_set()`): thread-local `{included_, excluded_}`. Modes inject themselves here (e.g., `Fake` added to `included_`).
3. **Dispatch keyset** (`dispatchKeySet`): computed ONCE at the start of a fresh dispatch via `computeDispatchKeySet`: `(tensor_keyset | tls.included_) - tls.excluded_`. Then threaded through handlers unchanged.
**Critical distinction — `callBoxed` vs `redispatchBoxed`:**
- `callBoxed` (fresh dispatch): recomputes keyset from tensor keysets + TLS via `getDispatchKeySetBoxed` → `computeDispatchKeySet`.
- `redispatchBoxed(ks, stack)`: uses the passed-in `ks` DIRECTLY — does NOT recompute from TLS. `entry.lookup(ks)` finds the highest-priority key in `ks` and calls that kernel.
**Why both modifications are needed in `fakeFallback`:**
- Modifying the `ks` passed to `redispatchBoxed`: handles THIS immediate redispatch (routes to Meta kernel).
- Modifying TLS (adding Meta to included, Fake to excluded): handles any SUB-dispatches during meta kernel execution (those are fresh `callBoxed` calls that recompute from TLS).
**`_force_tls_local_dispatch_key_set`:** Overwrites the entire TLS keyset (both `included_` and `excluded_`) unconditionally. No delta logic. `ForceDispatchKeyGuard` snapshots old TLS on construction, calls `_force_tls_local_dispatch_key_set(saved)` on destruction.

## Output Wrapping

## View / Aliasing

## Factory Ops

## Dispatch Key Ordering

### 2026-03-24: Fake must be BELOW Python in dispatch priority
**Context:** The Fake dispatch key was originally above Python in the `DispatchKey.h` enum. This meant `fakeFallback` fired before any TorchDispatchMode (ProxyTorchDispatchMode, FakeTensorMode), which are dispatched through the Python key. This broke composability with `make_fx` tracing.
**Fix:** Moved Fake before Python in the enum (lower enum value = lower priority). Removed `has_python_key_arg` function and its check — no longer needed since Python naturally fires first. Kept Python+PythonTLSSnapshot exclusion in the inner meta kernel redispatch (prevents meta kernel internal ops from triggering TorchDispatchModes, equivalent to Python's `_DisableTorchDispatch()`).
**Rule:** The C++ implementation must preserve the same dispatch ordering as the Python implementation. Python FakeTensorMode is a TorchDispatchMode (Python key level), so C++ Fake must be at or below Python priority.

## Python Key Bypass

## Integration with make_fx / Proxy Tracing

### 2026-03-26: extract_val creates conflicting Python FakeTensorMode when C++ mode active
**Symptom:** `test_make_fx_varargs` crashes with `RuntimeError: FakeTensor does not support meta device inputs` at `fake_impls.py:238` → `FakeFallbackKernel.cpp:84`.
**Root cause:** `extract_val` (`proxy_tensor.py:665`) calls `detect_fake_mode(val)` on a C++ fake tensor. Since C++ fake tensors aren't `FakeTensor` instances and there's no Python `FakeTensorMode` on the mode stack, it returns `None`. `extract_val` then creates a temp Python `FakeTensorMode` and calls `torch.empty_strided(device=val.device)`. The Python mode's `constructors` impl rewrites device to `meta`, then calls the op — but the C++ Fake dispatch key is still active in TLS, so `rewrite_device_args_to_meta` in `FakeFallbackKernel.cpp` rejects `meta` device inputs.
**Three Python functions unaware of C++ fake tensors:**
1. `is_fake` (`fake_tensor.py:202`): Only `isinstance(x, FakeTensor)` — C++ fake tensors are plain `torch.Tensor`
2. `detect_fake_mode` (`_guards.py:1330`): Checks Python `FakeTensorMode` on mode stack and `FakeTensor` instances only
3. `extract_val` (`proxy_tensor.py:665`): Falls through to creating temp Python `FakeTensorMode` when `detect_fake_mode` returns `None`
**Fix direction:** These functions need to detect C++ fake tensors via `torch._C._is_fake_tensor()` or `FakeTensorModeTLS` checks. Simplest: in `extract_val`, if `torch._C._is_fake_tensor(val)` is true, return `val` directly (it already has correct metadata).

## Other
