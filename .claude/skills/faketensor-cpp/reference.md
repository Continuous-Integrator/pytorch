# FakeTensor C++ — Architecture Reference

Read this file when you need detailed architecture context for implementation or debugging.

## Python Ground Truth

The Python FakeTensor lives in `torch/_subclasses/fake_tensor.py` (~3560 lines). Key classes:

- **`FakeTensor`** (line ~701): A `Tensor` subclass wrapping a meta tensor with a `fake_device` field. The `.device` property returns `meta` when `in_kernel_invocation` is True, otherwise the fake device. Constructed via `Tensor._make_subclass(..., device_for_backend_keys=device)` which sets backend dispatch keys (CUDA, autograd) to match the fake device while storage stays on meta.
- **`FakeTensorMode`** (line ~1297): A `TorchDispatchMode` that intercepts ops via `__torch_dispatch__`. Contains the dispatch cache, shape_env, converter, and `in_kernel_invocation` flag.
- **`FakeTensorConverter`** (line ~318): Manages conversion of real tensors to FakeTensors with memoization and storage aliasing. Uses `MetaConverter` internally.

## C++ Implementation Files

| File | Purpose |
|---|---|
| `aten/src/ATen/FakeFallbackKernel.cpp` | Core dispatch: `fakeFallback` boxed kernel, `get_common_device`, `rewrite_device_args_to_meta`, `wrap_outputs`, `transmute_to_fake` |
| `c10/core/impl/FakeTensorModeTLS.h/.cpp` | Thread-local state: activates/deactivates `Fake` + `Meta` dispatch keys in TLS |
| `c10/core/TensorImpl.h` | `FakeTensorMode` struct (line ~241), `ExtraMeta` with `fake_device_` and `fake_tensor_mode_` (line ~252) |
| `torch/csrc/Module.cpp` (lines ~2915-2982) | Python bindings: `_is_fake_tensor`, `_make_fake_tensor`, `_create_and_enter_fake_tensor_mode`, `_exit_fake_tensor_mode` |
| `test/test_faketensor_cpp.py` | Test suite using `cpp_fake_tensor_mode()` context manager |

## Dispatch Flow

```
torch op called
  → PythonDispatcher (highest priority, if active)
  → PreDispatch
  → Python key (__torch_dispatch__ / subclass dispatch)
       → TorchDispatchModes fire here (ProxyTorchDispatchMode, FakeTensorMode, etc.)
       → If Python FakeTensorMode handles the op, Fake key never fires
  → Fake key → fakeFallback (FakeFallbackKernel.cpp)
       → get_common_device from fake inputs
       → for factory ops: rewrite Device args to meta
       → exclude Fake+Python+PythonTLSSnapshot, redispatch
       → Meta kernel runs (shape-only computation)
       → wrap_outputs: stamp fake_device + FakeTensorMode on outputs
  → BackendSelect
  → Dense/Meta backend kernel
```

**Key invariant:** Fake is BELOW Python in dispatch priority. This means TorchDispatchModes
(including ProxyTorchDispatchMode for make_fx tracing) always fire before fakeFallback.
When the Python FakeTensorMode is active, it handles the op entirely and Fake never fires.
When only the C++ fake mode is active (no Python FakeTensorMode), fakeFallback handles it.

## TLS Activation

`FakeTensorModeTLS::set_state(mode)`:
- Adds `Fake` to TLS included dispatch keys
- Ops hit `fakeFallback` after any Python TorchDispatchModes have fired

`FakeTensorModeTLS::reset_state()`:
- Removes `Fake` from TLS included

## Python _dispatch_impl Priority Order

`FakeTensorMode._dispatch_impl()` (line ~2440 in `fake_tensor.py`) tries these in order:
1. `_DISPATCH_META_HANDLERS` — lightweight handlers for `prim.device`, `aten.size`, etc.
2. `_DISPATCH_HANDLE_DIRECTLY` — ops like `aten.is_coalesced`
3. Constant propagation shortcut
4. All-constant propagation
5. HOP (HigherOrderOperator) fast path
6. Fast op impls (`fake_impls.py`)
7. Python meta table (`torch._decomp.meta_table`)
8. Python decompositions (`torch._decomp.decomposition_table`)
9. `func.decompose()` for CompositeImplicitAutograd
10. prims meta impls
11. Custom op fake kernels
12. `op_implementations_checks` from `fake_impls.py`
13. Unsafe fallback
14. Meta kernel invocation via `in_kernel_invocation_manager`
15. `wrap_meta_outputs_with_default_device_logic()`

The C++ path currently handles steps 14-15 only. Everything else falls through to Python via the Python key bypass.

## Common Device Rules (Python `_find_common_device`)

- First FakeTensor sets `common_device`
- CPU 0-dim tensors are treated like scalars — non-CPU device wins
- If only CPU 0-dim seen so far, a non-zero-dim tensor of any device overrides
- `aten.nextafter.default` bypasses the zero-dim CPU check
- `aten._foreach_copy.default` allows CPU+non-CPU mixed device
- Any other device mismatch raises `RuntimeError`

## Dispatch Key Priority (high to low)

```
PythonDispatcher > PreDispatch > PythonTLSSnapshot > ... > AutogradCPU/CUDA >
ADInplaceOrView > ... > Python > Fake > BackendSelect > Dense(Meta/CPU/CUDA)
```

## DispatchKeySet Computation

```
effective_keys = ((tensor_keys | tls_included) - tls_excluded) & op_key_mask
```
Highest bit in `effective_keys` determines which kernel fires.

## Boxing

Fallback kernels (including `fakeFallback`) are always **boxed** — they receive an `IValue` stack and `OperatorHandle`. They use `op.redispatchBoxed(newKeySet, stack)` to continue dispatch.

## Integration with make_fx / Proxy Tracing

`make_fx` (`torch/fx/experimental/proxy_tensor.py`) traces a function by executing it and recording aten ops into an FX graph. Key components:

### Call Flow

```
make_fx(fn, tracing_mode)(*args)
  → _MakefxTracer.trace(fn, *args)
    → _init_modes_from_inputs():
        detect_fake_mode(args) → reuse or create FakeTensorMode
        _construct_modes_with_fx_tracer() → build ProxyTorchDispatchMode
    → _trace_inner(fn, *args):
        _wrap_fake(args)  — "fake"/"symbolic": real→FakeTensor; "real": identity
        Enter mode stack: FakeTensorMode → ProxyTorchDispatchMode
        dispatch_trace() → PythonKeyTracer.trace() → fn(*args) executes
```

### Op Recording

Every aten op hits `ProxyTorchDispatchMode.__torch_dispatch__` → `proxy_call()`:
1. Creates FX node: `tracer.create_proxy("call_function", func, ...)`
2. Executes op (produces FakeTensor if FakeTensorMode active)
3. `track_tensor_tree(output, proxy_output)` binds tensors to proxy nodes

### Metadata Attachment

```
track_tensor_tree → wrap_with_proxy → set_meta(proxy, tensor) → extract_val(tensor)
```

**`extract_val`** (`proxy_tensor.py:650`):
- If `is_fake(tensor)`: `snapshot_fake(tensor)` — detached copy
- If real tensor: `detect_fake_mode(tensor)` → create `FakeTensorMode` if None → `torch.empty_strided(shape, stride, device, dtype)` under it

**`detect_fake_mode`** (`torch/_guards.py:1330`): checks TracingContext → dispatch mode stack → FakeTensor inputs. Returns `None` if no Python FakeTensorMode found.

### Tracing Modes

| Mode | FakeTensorMode? | `extract_val` path |
|---|---|---|
| `"real"` | No | Creates temp `FakeTensorMode`, calls `empty_strided` |
| `"fake"` | Yes (static) | `snapshot_fake` |
| `"symbolic"` | Yes (ShapeEnv) | `snapshot_fake` |

### C++ Fake Tensor Integration Points

When `cpp_fake_tensor_mode()` is active, tests use `make_fx(fn, tracing_mode="real")`. The following Python functions are unaware of C++ fake tensors:

1. **`is_fake`** (`fake_tensor.py:202`): Only checks `isinstance(x, FakeTensor)`. C++ fake tensors are plain `torch.Tensor` → returns `False`.
2. **`detect_fake_mode`** (`_guards.py:1330`): Checks for Python `FakeTensorMode` on mode stack and `FakeTensor` instances. No C++ mode awareness → returns `None`.
3. **`extract_val`** (`proxy_tensor.py:665`): When `detect_fake_mode` returns `None`, creates a temp Python `FakeTensorMode` and calls `empty_strided` under it. This conflicts with the active C++ Fake dispatch key in TLS — the Python mode's `constructors` impl rewrites device to `meta`, then the C++ `rewrite_device_args_to_meta` rejects `meta` device inputs.

**To integrate:** These three functions need to detect C++ fake tensors (via `torch._C._is_fake_tensor()` or checking `FakeTensorModeTLS`) and handle them without creating a Python `FakeTensorMode`.
