---
name: faketensor-cpp
description: Help implement and debug the C++ FakeTensor rewrite in PyTorch. Use when working on FakeTensor C++ code, debugging fake tensor dispatch issues, understanding the dispatcher call stack, or comparing C++ behavior against the Python FakeTensor ground truth. Triggers on mentions of FakeTensor, fake tensor mode, FakeFallbackKernel, dispatch keys, or meta tensor conversion.
---

# FakeTensor C++ Implementation Assistant

You are helping rewrite PyTorch's FakeTensor from Python to C++. The Python implementation is the **ground truth**.

## Rules

1. **Python is always correct.** Never modify `torch/_subclasses/fake_tensor.py` to fix bugs. If behavior diverges, the C++ code is wrong.
2. **Read the actual source files.** Always read the real Python and C++ source code directly — do not rely solely on [reference.md](reference.md) or this skill's summaries. The reference file is a quick-start orientation; the source of truth is the code itself. Search the codebase when needed (grep for function names, dispatch keys, error messages, etc.).
3. **Consult past learnings first.** At the start of any debugging session, read [learnings.md](learnings.md). Past bugs recur.
4. **Record findings after.** After resolving any bug, append a dated entry to [learnings.md](learnings.md). Include symptom, root cause, and fix. Do not skip this.

## Key Files

- **Python ground truth:** `torch/_subclasses/fake_tensor.py`
- **C++ core dispatch:** `aten/src/ATen/FakeFallbackKernel.cpp`
- **C++ TLS:** `c10/core/impl/FakeTensorModeTLS.h`, `c10/core/impl/FakeTensorModeTLS.cpp`
- **C++ data structs:** `c10/core/TensorImpl.h` (`FakeTensorMode` struct, `ExtraMeta`)
- **Python bindings:** `torch/csrc/Module.cpp`
- **Tests:** `test/test_faketensor_cpp.py`
- **Full architecture details:** [reference.md](reference.md)

## Workflow: Debugging

1. **Read [learnings.md](learnings.md)** — check if this is a known pattern.
2. **Reproduce under both modes** — run the failing op under Python `FakeTensorMode` and C++ `cpp_fake_tensor_mode()`. Compare output shape, stride, dtype, device, and errors.
3. **Read the Python dispatch path** — open `torch/_subclasses/fake_tensor.py` and trace through `FakeTensorMode._dispatch_impl()` to understand what Python does for this op. Follow call chains into other files (`fake_impls.py`, `meta_utils.py`, etc.) as needed. [reference.md](reference.md) has a summary of the priority order, but always verify against the actual code.
4. **Read the C++ dispatch path** — open `aten/src/ATen/FakeFallbackKernel.cpp` and trace through `fakeFallback`. Search the codebase for related functions if the bug spans files (e.g., grep for dispatch key names, TLS helpers, TensorImpl methods). Check these functions in order:
   - `has_python_key_arg` — should it bail to Python?
   - `get_common_device` — does the device match Python's `_find_common_device`?
   - `rewrite_device_args_to_meta` — factory ops need device rewriting
   - `wrap_outputs` / `transmute_to_fake` — output stamping correct?
5. **Fix the C++ code only.** Never the Python.
6. **Write a test** in `test/test_faketensor_cpp.py` that covers the fix.
7. **Update [learnings.md](learnings.md)** with what you found.

## Workflow: Implementing a New Feature

1. **Read the Python code first.** Open `fake_tensor.py` and find the corresponding logic. Follow imports and call chains into other files as needed — don't stop at summaries.
2. **Decide if it belongs in C++.** The C++ path handles the "meta kernel + wrap outputs" fast path. Python-specific paths (constant propagation, decompositions, HOPs, cache) stay in Python — ops needing those paths use the Python key bypass.
3. **Write a comparison test first** in `test/test_faketensor_cpp.py` — run under both Python and C++ modes, assert identical results.
4. **Implement in C++**, keeping it simple. Search the codebase for existing patterns (grep for similar dispatch key usage, TensorImpl APIs, etc.) rather than guessing.
